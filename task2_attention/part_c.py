"""
Task 2 Part C driver logic: C1 validation, C2-C5 evaluation of every KV-cache method,
and the C5 packed encode/decode benchmark (Qwen/Qwen2.5-1.5B-Instruct, FP16, SDPA).

Prompt format (identical for all methods): the staff `user_text` as the single user
turn of the tokenizer's chat template with add_generation_prompt=True (Qwen's default
system prompt is inserted by the template).  The PRE-QUESTION CONTEXT is every token
before the token that contains the first character of "QUESTION_BEGIN"; the QUESTION
SUFFIX is that token through the end of the assistant header.  Token lengths are the
actual encoded lengths.  Evidence tokens = tokens whose character span overlaps
[evidence_char_start, evidence_char_end) (evaluation only, never given to a policy).

Per prompt (one FP16 prefill of the context, shared by every method):
  1. budget = M_4(L_ctx), verified by actually packing the INT4 channel-wise cache;
  2. question-aware importance once (suffix run on the full FP16 cache, then cropped);
  3. for each method: build its context cache (eviction / pack->unpack / hybrid),
     teacher-forced pass over suffix + gold code (reference-answer NLL, logit KL vs
     FP16), crop back, then greedy free-running generation (max 16 new tokens) with the
     quantised layouts re-quantising every completed 64-token group as it appears.
  The question suffix and generated tokens are the SHARED decode-time FP16 component
  (`shared_decode_fp16_bytes`), reported separately and added in total live memory.

Answer extractor (fixed, deterministic): the first run of exactly five digits in the
decoded generation (`extract_code`); exact match = equality with the gold code.
No retention parameter is tuned (8 sink / 64 recent fixed by the handout), so no
development prompts are used.
"""

import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from common.runio import read_jsonl, run_resumable, write_json
from common.timing import timed
from . import kv_cache as kvc

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EOS_IDS = (151645, 151643)
MAX_NEW_TOKENS = 16
METHODS = ["fp16", "sink_recent", "question_aware", "int4_tokenwise", "int4_channelwise", "hybrid"]
EVICTION = {"sink_recent", "question_aware"}
QUANT = {"int4_tokenwise": "tokenwise", "int4_channelwise": "channelwise"}
CODE_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
CASES_FILE = "data/kv_retrieval/kv_retrieval_cases.jsonl.gz"


def extract_code(text: str) -> Optional[str]:
    m = CODE_RE.search(text)
    return m.group(1) if m else None


def load_cases(data_repo: str) -> List[dict]:
    cases = read_jsonl(Path(data_repo) / CASES_FILE)
    assert len(cases) == 180, f"expected 180 retrieval cases, got {len(cases)}"
    return cases


def _sync(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return None


def build_prompt(tokenizer, case: dict) -> dict:
    user = case["user_text"]
    msgs = [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])
    ref = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ref, "keys"):
        ref = ref["input_ids"]
    assert list(ref) == ids, "offset tokenisation differs from apply_chat_template ids"
    offs = enc["offset_mapping"]
    u0 = text.index(user)
    q_char = u0 + user.index("QUESTION_BEGIN")
    split = next(i for i, (s, e) in enumerate(offs) if e > q_char)
    e0, e1 = u0 + case["evidence_char_start"], u0 + case["evidence_char_end"]
    ev = [i for i, (s, e) in enumerate(offs) if s < e1 and e > e0]
    assert ev and max(ev) < split, "evidence must lie inside the pre-question context"
    gold = tokenizer(case["code"], add_special_tokens=False)["input_ids"]
    return {"ids": ids, "context_len": split, "suffix_len": len(ids) - split, "total_len": len(ids),
            "evidence_tokens": ev, "gold_ids": gold}


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row from logits (fp32)."""
    lp = F.log_softmax(p_logits.float(), -1)
    lq = F.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def _model_dims(model) -> dict:
    c = model.config
    d_h = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return {"n_layers": c.num_hidden_layers, "n_kv": c.num_key_value_heads, "d_h": d_h,
            "n_heads": c.num_attention_heads}


@torch.no_grad()
def eval_prompt(model, tokenizer, case: dict, device: str, methods=METHODS,
                max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    dims = _model_dims(model)
    nl, nkv, dh = dims["n_layers"], dims["n_kv"], dims["d_h"]
    P = build_prompt(tokenizer, case)
    Lc = P["context_len"]
    ids = torch.tensor([P["ids"]], device=device)
    ctx, suffix = ids[:, :Lc], ids[:, Lc:]
    gold = torch.tensor([P["gold_ids"]], device=device)
    ev = set(P["evidence_tokens"])

    # shared FP16 prefill of the pre-question context
    _reset_peak(device)
    base = DynamicCache()
    _sync(device)
    t0 = time.perf_counter()
    kvc.forward_with_positions(model, base, ctx, 0, device, logits_to_keep=1)
    _sync(device)
    prefill_ms = 1000 * (time.perf_counter() - t0)
    prefill_peak = _peak_mb(device)

    budget = kvc.int_bytes(Lc, nl, nkv, dh, 4)
    fp16_bytes = kvc.kv_cache_bytes_fp16(Lc, nl, nkv, dh)
    pos_bytes = kvc.fp16_position_bytes(nl, nkv, dh)

    need_imp = any(m in ("question_aware", "hybrid") for m in methods)
    importance, scoring_ms, imp_info = None, 0.0, {}
    if need_imp:
        _sync(device)
        t0 = time.perf_counter()
        importance, imp_info = kvc.question_importance(model, base, suffix, Lc)
        _sync(device)
        scoring_ms = 1000 * (time.perf_counter() - t0)

    out = {"id": case["id"], "target_tokens": case["target_tokens"], "position": case["position"],
           "distractors": case["distractors"], "code": case["code"],
           "context_len": Lc, "suffix_len": P["suffix_len"], "total_len": P["total_len"],
           "n_evidence_tokens": len(ev), "evidence_tokens": P["evidence_tokens"],
           "gold_ids": P["gold_ids"], "budget_bytes_int4_channelwise": budget,
           "fp16_context_bytes": fp16_bytes, "prefill_ms": prefill_ms, "prefill_peak_alloc_mb": prefill_peak,
           "scoring_ms": scoring_ms, "scoring_info": imp_info, "dims": dims, "methods": {}}
    ref_first = ref_tf = None

    for m in methods:
        _reset_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        rec: Dict = {"method": m}
        done = {"upto": None}
        layout = QUANT.get(m)
        if m == "fp16":
            mc = kvc.clone_cache(base)
            rec["bytes"] = fp16_bytes
        elif m in EVICTION:
            n = kvc.max_eviction_positions(budget, Lc, nl, nkv, dh)
            pos = (kvc.select_sink_recent(Lc, n) if m == "sink_recent"
                   else kvc.select_question_aware(Lc, n, importance))
            mc = kvc._evict_to_positions(base, pos)
            rec["bytes"] = kvc.eviction_bytes(len(pos), nl, nkv, dh)
            rec["bytes_breakdown"] = {"fp16_payload": len(pos) * pos_bytes, "indices": len(pos) * kvc.INDEX_BYTES}
            rec["retained_positions"] = pos
            rec["n_fp16_positions"] = len(pos)
        elif layout:
            packed = kvc.encode_cache(base, layout, 4)
            _sync(device)
            rec["encode_ms"] = 1000 * (time.perf_counter() - t0)
            rec["bytes"] = kvc.packed_nbytes(packed)
            assert rec["bytes"] == budget, (m, rec["bytes"], budget)
            t1 = time.perf_counter()
            mc = kvc.decode_cache(packed)
            _sync(device)
            rec["decode_ms"] = 1000 * (time.perf_counter() - t1)
            del packed
            done["upto"] = kvc.channelwise_split_lengths(Lc)[0]
            rec["n_fp16_tail"] = Lc - done["upto"]
        elif m == "hybrid":
            fp16_pos, hb = kvc.select_hybrid(Lc, importance, budget, nl, nkv, dh, bits_low=2)
            t1 = time.perf_counter()
            packed = kvc.encode_hybrid(base, fp16_pos, bits_low=2)
            _sync(device)
            rec["encode_ms"] = 1000 * (time.perf_counter() - t1)
            rec["bytes"] = packed.nbytes
            assert rec["bytes"] == hb["total"] and rec["bytes"] <= budget, (rec["bytes"], hb, budget)
            t1 = time.perf_counter()
            mc = kvc.decode_hybrid(packed)
            _sync(device)
            rec["decode_ms"] = 1000 * (time.perf_counter() - t1)
            del packed
            rec["bytes_breakdown"] = hb
            rec["fp16_positions"] = fp16_pos
            rec["n_fp16_positions"] = len(fp16_pos)
            rec["precision_allocation"] = {"fp16_positions": hb["n_fp16"], "int2_positions": hb["n_low_quantised"],
                                           "fp16_residual_of_low_part": hb["n_low_fp16_residual"]}
        else:
            raise ValueError(m)
        _sync(device)
        rec["build_ms"] = 1000 * (time.perf_counter() - t0)
        rec["preprocess_ms"] = rec["build_ms"] + (scoring_ms if m in ("question_aware", "hybrid") else 0.0)
        rec["realized_fraction_of_fp16"] = rec["bytes"] / fp16_bytes
        rec["within_budget"] = rec["bytes"] <= budget if m != "fp16" else None
        if m in ("int4_tokenwise", "int4_channelwise", "hybrid"):
            rec["reconstruction"] = kvc.reconstruction_error(base, mc)
        if m in EVICTION:
            rec["evidence_retained_frac"] = len(ev & set(rec["retained_positions"])) / len(ev)
        elif m == "hybrid":
            rec["evidence_retained_frac"] = len(ev & set(rec["fp16_positions"])) / len(ev)

        phys = kvc._seq_len(mc)
        # teacher-forced pass: suffix + gold code (no requantisation inside one forward)
        tf_in = torch.cat([suffix, gold], dim=1)
        o = kvc.forward_with_positions(model, mc, tf_in, Lc, device, logits_to_keep=gold.shape[1] + 1)
        tf_logits = o.logits[0].float()                       # (n_gold+1, V): last suffix .. before last gold
        lp = F.log_softmax(tf_logits[:-1], -1)
        tok_nll = -lp.gather(-1, gold[0].unsqueeze(-1)).squeeze(-1)
        rec["ref_answer_nll_sum"] = float(tok_nll.sum())
        rec["ref_answer_nll_mean"] = float(tok_nll.mean())
        mc.crop(phys)

        # free-running greedy generation
        def after_step(c, layout=layout):
            if layout:
                done["upto"] = kvc.requantize_completed_groups(c, layout, 4, done["upto"])

        _sync(device)
        t0 = time.perf_counter()
        gen, first = kvc.greedy_generate_edited(model, mc, suffix, Lc, max_new_tokens, EOS_IDS, device,
                                                after_step=after_step)
        _sync(device)
        rec["suffix_plus_decode_ms"] = 1000 * (time.perf_counter() - t0)
        n_gen = gen.shape[1]
        rec["decode_ms_per_token"] = rec["suffix_plus_decode_ms"] / max(1, n_gen)
        rec["generated_ids"] = gen[0].tolist()
        rec["generation"] = tokenizer.decode(gen[0], skip_special_tokens=True)
        rec["extracted"] = extract_code(rec["generation"])
        rec["exact_match"] = int(rec["extracted"] == case["code"])
        live = kvc._seq_len(mc)
        rec["live_cache_len_end"] = live
        rec["shared_decode_fp16_bytes"] = (live - phys) * pos_bytes
        rec["total_live_bytes"] = rec["bytes"] + rec["shared_decode_fp16_bytes"]
        if layout:
            rec["total_live_bytes_actual_requantised"] = kvc.int_bytes(Lc + (live - phys), nl, nkv, dh, 4)
        rec["peak_alloc_mb"] = _peak_mb(device)
        if m == "fp16":
            ref_first, ref_tf = first, tf_logits
            rec["post_prefill_kl"] = 0.0
            rec["tf_kl_mean"] = 0.0
        else:
            rec["post_prefill_kl"] = float(_kl(ref_first, first)[0])
            rec["tf_kl_mean"] = float(_kl(ref_tf, tf_logits).mean())
        rec["first_token_argmax"] = int(first.argmax(-1)[0])
        out["methods"][m] = rec
        del mc
    fp = out["methods"].get("fp16")
    if fp:
        for m, r in out["methods"].items():
            r["reference_correct"] = fp["exact_match"]
    del base
    return out


def load_model(device, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").to(device).eval()
    return model, tok


def run_eval(results_root: str, data_repo: str, device: str, overwrite=False, limit=None,
             model=None, tokenizer=None, revision=None, log=print) -> Dict[str, str]:
    root = Path(results_root)
    cases = load_cases(data_repo)
    if limit:
        cases = cases[:limit]
    if model is None:
        model, tokenizer = load_model(device, revision)
    by_id = {c["id"]: c for c in cases}

    def one(cid, out_dir):
        r = eval_prompt(model, tokenizer, by_id[cid], device)
        write_json(out_dir / "summary.json", r, indent=None)
        ms = r["methods"]
        log(f"[{cid}] L={r['context_len']} " + " ".join(f"{m}:{ms[m]['exact_match']}" for m in ms))

    return run_resumable([c["id"] for c in cases], lambda n: root / "prompts" / n, one,
                         overwrite=overwrite, log=log)


# ── C1 validation ─────────────────────────────────────────────────────────────

# Bump when the C1 pass criterion changes: run_task2 recomputes c1_validation.json whenever
# the stored "criterion_version" differs from this value.
C1_CRITERION_VERSION = 2
C1_NOTE = ("pass = all logits finite AND 100%-retention greedy tokens identical to HF greedy generate "
           "AND scoring leaves the cache unchanged AND the last suffix position's argmax agrees between "
           "one full forward and split prefill+suffix AND every argmax mismatch at an earlier suffix "
           "position is a near tie (full-forward top-1 minus top-2 margin <= 2 x that position's "
           "max |full - split| logit difference, a necessary condition for an FP16 rounding-induced flip). "
           "The raw split_vs_full_argmax_agreement is reported but is not required to be 1.0.")


def argmax_mismatch_report(full: torch.Tensor, split: torch.Tensor) -> dict:
    """Compare per-position argmax of two (Q, V) logit matrices (full forward vs split
    prefill).  For every mismatching position: the full-forward top-1 minus top-2 margin
    and the max |full - split| logit difference at that position."""
    full, split = full.float(), split.float()
    diff = (full - split).abs().max(-1).values                      # (Q,)
    top2 = full.topk(2, dim=-1).values                              # (Q, 2)
    margin = top2[:, 0] - top2[:, 1]
    agree = full.argmax(-1) == split.argmax(-1)
    mism = [int(i) for i in torch.nonzero(~agree).flatten().tolist()]
    details = [{"position": i, "full_top1_minus_top2_margin": float(margin[i]),
                "max_abs_logit_diff": float(diff[i]),
                "near_tie": bool(margin[i] <= 2 * diff[i])} for i in mism]
    return {"split_vs_full_argmax_agreement": float(agree.float().mean()),
            "mismatch_positions": mism, "mismatch_details": details,
            "all_mismatches_near_tie": all(d["near_tie"] for d in details),
            "last_position_argmax_agrees": bool(agree[-1])}


def quant_diagnostic(cache: DynamicCache, bits: int = 4, group_size: int = kvc.GROUP_SIZE) -> List[dict]:
    """Per layer: relative Frobenius error of K packed/unpacked token-wise and
    channel-wise, and of V token-wise (INT4, 64-element groups, the existing
    pack_tensor/unpack_tensor; the trailing L mod 64 tokens stay FP16 in both layouts),
    and the ratio of the largest to the median per-channel max |K| (channel = one
    (KV head, head-dim) coordinate, max over tokens)."""
    rows = []

    def rel(X, layout):
        Xh = kvc.unpack_tensor(kvc.pack_tensor(X, layout, bits, group_size), X.dtype, group_size).float()
        Xf = X.float()
        den = float(Xf.pow(2).sum())
        return math.sqrt(float((Xf - Xh).pow(2).sum()) / den) if den else 0.0

    for i in range(kvc._num_layers(cache)):
        K, V = kvc._layer_keys(cache, i), kvc._layer_values(cache, i)
        ch = K.float().abs().amax(dim=(0, 2)).flatten()             # (H_kv * d_h,)
        med = float(ch.median())
        rows.append({"layer": i,
                     "K_rel_fro_token": rel(K, "token"),
                     "K_rel_fro_channel": rel(K, "channel"),
                     "V_rel_fro_token": rel(V, "token"),
                     "K_channel_absmax_max_over_median": float(ch.max()) / med if med else float("inf")})
    return rows


@torch.no_grad()
def c1_validate(model, tokenizer, cases: List[dict], device: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """(i) full-forward vs split prefill/suffix logits (argmax mismatches must be near
    ties, the last position must agree); (ii) 100% retention through the eviction path
    reproduces ordinary greedy generation; (iii) all logits finite; (iv) the manual FP32
    scoring attention matches the SDPA attention output and leaves the cache unchanged.
    Also records a KV quantisation diagnostic (not part of pass) on each context prefill."""
    res, qd = [], []
    for case in cases:
        P = build_prompt(tokenizer, case)
        Lc, Q = P["context_len"], P["suffix_len"]
        ids = torch.tensor([P["ids"]], device=device)
        full = model(ids, use_cache=False, logits_to_keep=Q).logits[0].float()
        cache = DynamicCache()
        kvc.forward_with_positions(model, cache, ids[:, :Lc], 0, device, logits_to_keep=1)
        qd.append({"id": case["id"], "context_len": Lc, "layers": quant_diagnostic(cache)})
        k0 = kvc._layer_keys(cache, 0).clone()
        imp, info = kvc.question_importance(model, cache, ids[:, Lc:], Lc, validate=True)
        unchanged = kvc._seq_len(cache) == Lc and torch.equal(k0, kvc._layer_keys(cache, 0))
        split = kvc.forward_with_positions(model, kvc.clone_cache(cache), ids[:, Lc:], Lc, device).logits[0].float()
        diff = (full - split).abs()
        mm = argmax_mismatch_report(full, split)
        ref = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             repetition_penalty=1.0, eos_token_id=list(EOS_IDS),
                             pad_token_id=tokenizer.eos_token_id)[0, ids.shape[1]:].tolist()
        full_ret = kvc._evict_to_positions(cache, list(range(Lc)))
        got, first = kvc.greedy_generate_edited(model, full_ret, ids[:, Lc:], Lc, max_new_tokens, EOS_IDS, device)
        got = got[0].tolist()
        n = min(len(ref), len(got))
        res.append({
            "id": case["id"], "context_len": Lc, "suffix_len": Q,
            "split_vs_full_max_abs_logit_diff": float(diff.max()),
            **mm,
            "all_logits_finite": bool(torch.isfinite(full).all() and torch.isfinite(split).all()
                                      and torch.isfinite(first).all()),
            "full_retention_tokens": got, "hf_greedy_tokens": ref,
            "full_retention_matches_greedy": got[:n] == ref[:n] and len(got) == len(ref),
            "scoring_cache_unchanged": bool(unchanged),
            "scoring_manual_vs_sdpa_max_abs_err": max(info["manual_vs_sdpa_attention_max_abs_err_per_layer"]),
        })
    for r in res:
        r["pass"] = bool(r["all_logits_finite"] and r["full_retention_matches_greedy"]
                         and r["scoring_cache_unchanged"] and r["last_position_argmax_agrees"]
                         and r["all_mismatches_near_tie"])
    return {"criterion_version": C1_CRITERION_VERSION, "cases": res, "pass": all(r["pass"] for r in res),
            "note": C1_NOTE,
            "quant_diagnostic": {"bits": 4, "group_size": kvc.GROUP_SIZE, "cases": qd,
                                 "note": "diagnostic only, not part of pass: relative Frobenius error of "
                                         "pack->unpack over the whole (B,H,L,d) tensor of each layer after "
                                         "the FP16 context prefill (trailing L mod 64 tokens are FP16 in "
                                         "both layouts); int4_tokenwise uses K token + V token, "
                                         "int4_channelwise uses K channel + V token"}}


# ── C5 packed encode/decode benchmark ─────────────────────────────────────────

BENCH_LAYOUTS = {"int4_tokenwise": ("tokenwise", 4), "int4_channelwise": ("channelwise", 4),
                 "int2_channelwise_hybrid_lowbit": ("channelwise", 2)}


@torch.no_grad()
def bench_packed(model, device: str, batches=(1, 8), lengths=(2048, 4096), n_warmup=3, n_rep=10,
                 log=print) -> List[dict]:
    """Packing / unpacking time, serialized bytes and decode-step latency with
    unpacking vs an FP16 reference cache at matched shapes (random FP16 K/V)."""
    dims = _model_dims(model)
    nl, nkv, dh = dims["n_layers"], dims["n_kv"], dims["d_h"]
    rows = []
    for B in batches:
        for L in lengths:
            g = torch.Generator(device="cpu").manual_seed(6304 + B * 10007 + L)
            layers = [(torch.randn(B, nkv, L, dh, generator=g).to(device, torch.float16),
                       torch.randn(B, nkv, L, dh, generator=g).to(device, torch.float16)) for _ in range(nl)]
            tok = torch.full((B, 1), min(151643, model.config.vocab_size - 1), dtype=torch.long, device=device)
            pos = torch.full((B, 1), L, dtype=torch.long, device=device)
            cpos = torch.tensor([L], device=device)

            ref_cache = kvc.cache_from_tensors([(k.clone(), v.clone()) for k, v in layers])

            def fp16_step():
                model(input_ids=tok, past_key_values=ref_cache, position_ids=pos, cache_position=cpos,
                      use_cache=True, logits_to_keep=1)
                ref_cache.crop(L)

            _reset_peak(device)
            t_ref = timed(fp16_step, n_warmup=n_warmup, n_rep=n_rep)
            ref_peak = _peak_mb(device)
            del ref_cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            base = kvc.cache_from_tensors(layers)
            for name, (layout, bits) in BENCH_LAYOUTS.items():
                holder = {}

                def enc():
                    holder["p"] = kvc.encode_cache(base, layout, bits)

                t_enc = timed(enc, n_warmup=n_warmup, n_rep=n_rep)
                packed = holder["p"]
                nbytes = kvc.packed_nbytes(packed)
                formula = B * kvc.int_bytes(L, nl, nkv, dh, bits)

                def dec():
                    holder["c"] = kvc.decode_cache(packed)

                t_dec = timed(dec, n_warmup=n_warmup, n_rep=n_rep)
                holder.pop("c", None)

                def unpack_step():
                    c = kvc.decode_cache(packed)
                    model(input_ids=tok, past_key_values=c, position_ids=pos, cache_position=cpos,
                          use_cache=True, logits_to_keep=1)

                _reset_peak(device)
                t_step = timed(unpack_step, n_warmup=n_warmup, n_rep=n_rep)
                rows.append({"layout": name, "bits": bits, "B": B, "L": L,
                             "serialized_bytes": nbytes, "formula_bytes": formula,
                             "bytes_match_formula": nbytes == formula,
                             "fp16_bytes": B * kvc.kv_cache_bytes_fp16(L, nl, nkv, dh),
                             "pack_ms": t_enc, "unpack_dequant_ms": t_dec,
                             "decode_step_with_unpack_ms": t_step, "decode_step_fp16_ms": t_ref,
                             "peak_alloc_mb_unpack_step": _peak_mb(device), "peak_alloc_mb_fp16_step": ref_peak,
                             "n_warmup": n_warmup, "n_rep": n_rep,
                             "note": "decode step = one batched token through the full model; the unpacked "
                                     "cache is dequantised to FP16 before ordinary SDPA"})
                log(f"[bench] {name} B={B} L={L}: pack {t_enc['mean_ms']:.2f} ms unpack {t_dec['mean_ms']:.2f} ms "
                    f"step {t_step['mean_ms']:.2f} vs fp16 {t_ref['mean_ms']:.2f} ms, bytes {nbytes} == {formula}")
                del packed
                holder.clear()
            del base, layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows
