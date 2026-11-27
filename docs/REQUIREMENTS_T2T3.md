# Requirements checklist — PA4 §2 (Task 2) and §3 (Task 3)

Source: `ATML_PA4.pdf` §2 pp. 5–11 and §3 pp. 11–15, read with `pdftotext -layout`.
One line per requirement. **Status** is the state of the repo *before* the Phase 1–3 work
(commit `72bfd07`): an existing `path::function` that covers it, `PARTIAL (path: reason)`,
or `MISSING`. The final mapping to code (path:line) is in the session's final report and
in `GAPS.md`.

Optional items: §2/§3 mark **nothing** as optional. The only non-required material is
the "Additional" reading lists (§2 p.7, §3 p.12) and the cross-task discussion (§4,
"not required"). Both are skipped: optional.

Data locations (Colab): `/content/drive/MyDrive/atml-pa4-datasets/data/{mqar,kv_retrieval,short_form,public_locked,prefix_banks}`,
generator `scripts/mqar_generator.py` in the same data repo.

## Task 2 Part A — attention operators

| ID | Requirement | Status (pre-work) |
|---|---|---|
| A-0 | Identical synthetic Q,K,V; B=2, H=4, d_h=64 | MISSING (no sweep driver) |
| A-0b | L ∈ {256, 512, 1024, 2048, 4096, 8192} | MISSING |
| A-0c | FP16 and FP32 "as supported" | MISSING |
| A-0d | Outputs **and gradients** vs dense FP32 reference at feasible lengths; larger shapes benchmarked without forcing infeasible dense | MISSING |
| A-0e | 10 warmed-up repetitions per supported condition | MISSING (`common/timing.py::timed` exists, default n_rep=5) |
| A-0f | Record the actual SDPA backend, do not assume | PARTIAL (`attention_ops.py::_detect_sdpa_backend` reports *enabled* flags, not the dispatched kernel) |
| A1-1 | Dense: explicit QKᵀ, causal mask, stable softmax, ×V | `attention_ops.py::dense_causal_attention` |
| A1-2 | Verify causality by perturbing future keys/values | PARTIAL (`attention_ops.py::verify_causality` checks only position 0) |
| A1-3 | Output error and backward-gradient error vs FP32 reference | MISSING |
| A1-4 | All-masked / extreme-score edge cases | MISSING |
| A2-1 | Exact blockwise online-softmax forward, no full score matrix | PARTIAL (`attention_ops.py::tiled_causal_attention`: FP16 accumulators, in-place writes, no all-masked guard) |
| A2-2 | Verify across tile sizes and dtypes | MISSING |
| A2-3 | Numerical stability and intermediate storage | MISSING |
| A2-4 | Forward/backward timing vs dense | MISSING |
| A2-5 | State that the Python tiled backward is autograd and stores intermediates; not FlashAttention | PARTIAL (docstring only; must be written into results) |
| A3-1 | SDPA on identical tensors with dispatched/forced backend recorded | PARTIAL (`attention_ops.py::sdpa_attention`, deprecated `sdp_kernel`, no forcing) |
| A3-2 | Forward latency; full forward+backward time | MISSING |
| A3-3 | Peak allocated **and reserved** GPU memory | MISSING |
| A3-4 | Output/gradient agreement | MISSING |
| A3-5 | Unsupported backends / OOM noted explicitly | MISSING |
| A-E1 | Evidence: output+gradient error by dtype/length | MISSING |
| A-E2 | Evidence: latency vs length; allocated-memory curves | MISSING |
| A-E3 | Evidence: backend/unsupported-operation table; fwd-bwd distinguished from fwd-only | MISSING |

## Task 2 Part B — MQAR

| ID | Requirement | Status (pre-work) |
|---|---|---|
| B-M1 | 2 pre-norm blocks, hidden 128, 4 heads × 32, 4× MLP, learned token emb, sinusoidal PE, vocab 256 | `mqar_model.py::MQARTransformer` |
| B-M2 | Identical projections/norm/optimizer/budget; only the attention operator changes | `mqar_model.py::TransformerBlock` |
| B-M3 | K classification logits at query positions; CE vs value ID at same position, others masked | `mqar_model.py::MQARTransformer.forward`, `mqar_train.py::train_mqar` |
| B-T1 | AdamW peak LR 8e-4, wd 0.01 | `mqar_train.py::train_mqar` |
| B-T2 | Batch 32, 4,500 updates | `mqar_train.py::train_mqar` |
| B-T3 | 200 warmup, linear decay to 15% of peak | `mqar_train.py::linear_warmup_decay` (check: decays to 15% at step 4500) |
| B-T4 | Global grad-norm clip 1.0 | `mqar_train.py::train_mqar` |
| B-T5 | Checkpoint eval at 500, 1000, 1500, 2000, 3000, 4500 | `mqar_train.py::CHECKPOINT_STEPS` |
| B-T6 | Seeds 6304, 6305, 6306 for every trained architecture and intervention | MISSING (no grid driver) |
| B-T7 | Preserve learning curves and the final checkpoint | PARTIAL (`train_mqar` writes to repo `results/`, not Drive) |
| B-D1 | Use the supplied MQAR generator (T=256), verify invariant checks before training | MISSING (no loader; generator lives in data repo) |
| B-D2 | Keep supplied disjoint train / dev / final-test streams | MISSING |
| B-D3 | Save source pos, query pos, answer, distance per query | supplied data (fields present) |
| B-E1 | 1,024 final-test sequences per (K, regime), accuracy over supervised query positions | PARTIAL (`mqar_train.py::evaluate_mqar`, no per-cell breakdown) |
| B-E2 | Chance baselines 1/64 and 1/K | MISSING |
| B-E3 | Accuracy for distance bins [0,32), [32,64), [64,128), [128,256) and the three regimes | MISSING |
| B-E4 | Report the 2(W−1)=126 architectural access limit for two-layer W=64 | MISSING |
| B-E5 | Use the saved source position (not query position) for distance | supplied `distances` field; MISSING in eval aggregation |
| B1-1 | Dense K=4,8,16 mixed | MISSING (grid) |
| B1-2 | Dense K=16 near-only control | MISSING |
| B1-3 | Window W=64 K=4,8,16 near-only | MISSING; **bug**: `SlidingWindowAttention` builds its bias as `mask.float()*-inf`, giving NaN (0·−∞) on allowed entries |
| B1-4 | Window W=64 K=16 mixed | MISSING |
| B1-5 | Evaluate every model on the same near/far/mixed test sets | MISSING |
| B1-6 | Test-time-only W=64 mask on the learned dense model vs the trained window model | MISSING |
| B1-7 | Preserve all training curves; report failure to learn | PARTIAL |
| B2-1 | Causal ELU+1 linear attention K=4,8,16 mixed | `mqar_model.py::LinearAttention` (Python loop over T; very slow for training) |
| B2-2 | Mixed local/dilated K=4 and 16: 2 heads W=64, 2 heads 64 positions at offsets {0,4,…,252} | PARTIAL (`MixedDilatedAttention`: same NaN mask bug; offsets not capped at 64 for T>256) |
| B2-3 | Same head count, hidden, depth | `mqar_model.py` |
| B2-4 | Compare near/far accuracy, loss, runtime, candidate positions per query | MISSING |
| B2-5 | Dense boolean mask acceptable but not described as an efficient sparse kernel | MISSING (must be stated in results) |
| B3-1 | One intervention motivated by observed results, K=16, all three seeds, 4,500 updates | MISSING |
| B3-2 | Within 5% of baseline trainable parameter count | MISSING |
| B3-3 | Final-test sets unchanged | MISSING |
| B3-4 | Prediction stated before evaluation | MISSING |
| B3-5 | Report extra persistent state, attended positions/state dims, dev-selection rule | MISSING |
| B3-6 | Persistent compressed state: total state bytes/layer (incl. local buffer) matched to the linear comparator; document precision/feature adjustment | MISSING |
| B-G | Grid = (8 B1 + 5 B2 + 1 B3) configs × 3 seeds = **42 runs** | MISSING |
| B-R1 | Theoretical attention pair counts and state bytes at T=256 | MISSING |
| B-R2 | Forward-only sweep T ∈ {256,1024,4096,8192}, fixed random embeddings, matched B,H,d | MISSING |
| B-R3 | Report explicitly when an implementation materializes dense masks / linear prefix tensors | MISSING |
| B-Ev1 | Learning curves for every trained mechanism | MISSING (plots) |
| B-Ev2 | near/far/mixed accuracy by number of pairs | MISSING |
| B-Ev3 | Distance-by-load heatmap identifying training K | MISSING |
| B-Ev4 | Train-vs-test mask controls | MISSING |
| B-Ev5 | Separate theoretical vs measured resource costs | MISSING |

## Task 2 Part C — pretrained KV memory (Qwen2.5-1.5B-Instruct, FP16)

| ID | Requirement | Status (pre-work) |
|---|---|---|
| C-0a | Greedy generation; same prompt format for all policies | PARTIAL (`kv_cache.py::greedy_generate_edited`) |
| C-0b | Staff retrieval JSONL + gold spans; do not regenerate/relabel | MISSING (no loader) |
| C-0c | 180 prompts = 2 lengths × 3 positions × 3 distractor counts × 10 | MISSING |
| C-0d | Measure and retain actual chat-template token lengths | MISSING |
| C-0e | Fixed deterministic answer extractor | MISSING |
| C-0f | Importance policies never see the code or gold coordinates | `kv_cache.py::_score_positions_by_attention` (does not read them) |
| C-0g | Dev prompts for retention params distinct from final test | n/a — no parameters are tuned (8 sink, 64 recent fixed by PDF); must be stated |
| C-B1 | One case-specific budget = actual serialized bytes of uniform INT4 channel-K/token-V reference | PARTIAL (`common/byte_accounting.py::kv_cache_bytes_int`) |
| C-B2 | Group 64, FP16 scale + FP16 offset (4 B), packed payload; zero-point handling reported | PARTIAL (`kv_cache.py::_quantize_groups`, fake-quant only, no packing) |
| C-B3 | Only complete 64-token groups compressed; trailing L mod 64 FP16 for K and V, incl. new tokens | PARTIAL (`quantize_kv_channelwise_k`, prompt only; no decode-time path) |
| C-B4 | Budget measured where the pre-question context is compressed | MISSING |
| C-B5 | Question suffix + generated FP16 tokens reported as a separate shared decode-time component and included in total live memory | MISSING |
| C-B6 | Packing/padding, indices, FP16 residual in every total | PARTIAL |
| C-B7 | Eviction: largest legal number of FP16 positions ≤ budget | PARTIAL (`sink_plus_recent_eviction`, `question_aware_eviction`) |
| C-B8 | Hybrid: same accounting (low-bit groups, metadata, indices, FP16) | PARTIAL (`hybrid_kv`: greedy break, V tail not FP16, low part padded) |
| C-B9 | Realized fraction of full FP16 bytes per prompt | MISSING |
| C1-1 | Full-forward logits vs split prefill/decode logits | PARTIAL (`tests/test_kv_cache.py::gpu_test_full_vs_split_prefill_decode`, never run) |
| C1-2 | 100% retention reproduces ordinary generation; logits finite | PARTIAL (`tests/test_kv_cache.py::gpu_test_full_retention_matches_greedy`, never run) |
| C1-3 | Preserve absolute RoPE positions; consistent physical cache length | `kv_cache.py::forward_with_positions` |
| C1-4 | Full-cache retrieval accuracy on every cell first | MISSING |
| C2-1 | Sink+recent FP16 (8 sink, 64 recent, fill with most recent) | `kv_cache.py::sink_plus_recent_eviction` |
| C2-2 | Question-aware FP16 (8 sink, 64 recent, fill by question-suffix attention) | `kv_cache.py::question_aware_eviction` |
| C2-3 | SDPA model kept; hook q_proj, RoPE, masked FP32 attention vs cached keys | PARTIAL (`_score_positions_by_attention` runs the suffix without the context, so layer ≥1 queries are wrong) |
| C2-4 | State shared vs per-head/layer retention and count elements accordingly | `kv_cache.py` module docstring (global) |
| C3-1 | Token-wise INT4 K and V, group 64 | PARTIAL (`quantize_kv_tokenwise`: no FP16 trailing rule) |
| C3-2 | Channel-wise-K / token-wise-V INT4, group 64 | `quantize_kv_channelwise_k` |
| C3-3 | Same rounding/metadata and same FP16 trailing rule for both; packed bytes verify budget | PARTIAL |
| C3-4 | K and V reconstruction errors separately | MISSING |
| C3-5 | Teacher-forced logit divergence | MISSING |
| C3-6 | Free-running answer accuracy | MISSING |
| C3-7 | Quantize prompt-cache entries and subsequently generated entries consistently | MISSING |
| C4-1 | Hybrid: 8 sink + 64 recent FP16, + question-aware FP16, rest channel-K/token-V INT2 in complete groups, residual FP16 | PARTIAL (`hybrid_kv`) |
| C4-2 | Choose #FP16 positions with all metadata/indices/padding counted; never exceed budget | PARTIAL |
| C4-3 | Compare hybrid with QA eviction and uniform channel-wise INT4 | MISSING |
| C4-4 | Report precision allocation and selection/reordering costs | MISSING |
| C5-1 | 5 compressed configs vs FP16 reference on identical prompts | MISSING |
| C5-2 | Exact-match, reference-answer NLL, post-prefill logit KL | MISSING |
| C5-3 | Fraction of marked evidence tokens retained (eviction + hybrid FP16 selection) | MISSING |
| C5-4 | Realized packed bytes, scoring/preprocessing, prefill, decode time, peak allocated memory | MISSING |
| C5-5 | Paired bootstrap over prompts, ≥2,000 resamples, principal differences | PARTIAL (`common/bootstrap.py::bootstrap_paired_accuracy`) |
| C5-6 | Packed encode/decode benchmark: batch 1 and 8, both context lengths, ≥10 warmed synchronized reps | MISSING |
| C5-7 | Packing time, unpack/dequant time, serialized bytes, decode-step latency with unpacking vs FP16 | MISSING |
| C5-8 | Primary differences over full paired set; per-cell results descriptive | MISSING |
| C5-9 | Distinguish packed storage from allocated GPU memory (dequantized before SDPA) | MISSING |
| C-W1 | Keep reference-wrong examples; also report accuracy conditional on reference-correct | MISSING |
| C-Ev1 | Single matched-INT4 table: 2 eviction, 2 INT4 layouts, hybrid, FP16 (acc, NLL, KL, evidence, bytes) | MISSING |
| C-Ev2 | Breakdowns by length, evidence position, distractor count; paired CIs | MISSING |
| C-Ev3 | Systems table: prefill/scoring/decode times; calculated vs serialized size vs allocated memory | MISSING |
| C-Ev4 | Small fixed set of aligned generations with evidence spans, FP16 selections, selection masks | MISSING |

## Task 3 — distillation and speculative decoding

| ID | Requirement | Status (pre-work) |
|---|---|---|
| S-M1 | Target/teacher Qwen2.5-1.5B-Instruct, draft/student Qwen2.5-0.5B-Instruct, frozen backbones, LoRA on student | `distill.py::_build_lora_student` |
| S-M2 | Check token IDs and chat template agree between models on a fixed prompt set | MISSING |
| S-M3 | Common non-thinking chat format, fixed EOS {151645, 151643} | PARTIAL (`speculative.py::QWEN_EOS_IDS`; prompts tokenized *without* the chat template in `distill.py` / `prefix_eval.py`) |
| S-D1 | 500 GSM8K train IDs + 300 short-form; no resampling | PARTIAL (`distill.py::_build_prompt_pool`) |
| S-D2 | 800-prompt pool, fixed seeded round-robin permutation within the 2 categories, reuse after exhaustion | PARTIAL (`_build_prompt_pool`: stops at 800, no reuse, no quota-driven schedule) |
| S-D3 | Off-policy: frozen teacher-response collection first, then cycle deterministically to quota | PARTIAL (`_collect_teacher_responses`: stores full-vocab logits as JSON lists — infeasible) |
| S-D4 | On-policy: fresh student responses at the current checkpoint, same prompt schedule | PARTIAL (`onpolicy_distill`) |
| S-D5 | Store source ID, rollout seed, response IDs, termination reason, #teacher-scored tokens per presentation | MISSING |
| S-D6 | GSM8K test + final short-form prompts excluded; held-out template families reserved | supplied split (`short_test.jsonl` familiar/shifted); MISSING check |
| S-B1 | Seeds 6304, 6305 × {off, on}, each from the original checkpoint | MISSING (driver) |
| S-B2 | Quota 40,000 scored tokens per method/seed: 30,000 math + 10,000 short | PARTIAL (only total quota enforced) |
| S-B3 | Caps 192 math / 32 short; temperature 1, top-p 1 (top-k 0, rep. penalty 1.0) | PARTIAL (top_k not set → Qwen generation_config top_k=20 applies; `generator=` unused) |
| S-B4 | Microbatch 1; one update per 256 teacher-scored tokens, short final update | PARTIAL (updates when ≥256, not exactly 256) |
| S-B5 | AdamW 3e-4, wd 0.01; FP16 frozen backbone; FP32 adapters/optimizer; FP16 autocast + dynamic grad scaling; unscale before clip | PARTIAL (`distill.py`) |
| S-B6 | Clip 1.0; KL normalized by supervised tokens per update | PARTIAL (`F.kl_div(batchmean)` then `/n*256` → normalised by n²/256) |
| S-B7 | LoRA r=8, α=16, dropout 0, q/k/v/o | `distill.py::LORA_CONFIG` |
| S-B8 | Record skipped/overflowed updates | PARTIAL (`isfinite(grad_norm)` check, not GradScaler-based) |
| S-B9 | Truncate at quota mid-response and record | MISSING |
| S-L1 | Forward KL(p_T ‖ q_φ) over full vocab averaged over supervised prefixes; no hard labels | PARTIAL (`distill.py::_kl_loss`) |
| S-L2 | Count teacher generation, teacher scoring, student rollouts, backward updates, merging separately | MISSING |
| S-E1 | 120 test prompts: 60 GSM8K + 60 short (30 familiar / 30 shifted) | MISSING (loader) |
| S-E2 | Identical prompts/format/stopping rules for all drafts | MISSING |
| S-E3 | Temperature 1, top-p 1; γ ∈ {2, 4}; two paired repetitions | MISSING |
| S-E4 | Primary EOS-aware speed benchmark with 64-new-token cap | MISSING |
| S-E5 | Separate answer-quality eval: 512 math / 32 extraction caps | MISSING |
| S-E6 | 24 math prompts fixed-64-token throughput diagnostic ignoring EOS for timing | MISSING |
| S-P1 | Supplied teacher and original-draft banks used | PARTIAL (`prefix_eval.py::evaluate_prefix_bank`, no chat template, wrong position semantics) |
| S-P2 | exact_spec_original_gamma4 bank generated by our sampler (not supplied) | PARTIAL (`prefix_eval.py::generate_exact_spec_bank`, wrong prefix length pos vs pos−1, no chat template) |
| S-P3 | Up to 32 positions from 1–64, bins [1,16],[17,32],[33,64]; separate 65–192 bank | supplied banks; MISSING for our own banks |
| S-P4 | Same saved prefix IDs evaluated for every draft | MISSING |
| S-P5 | Own-policy prefix banks for each trained student, producing checkpoint recorded, cross-evaluated | MISSING |
| S1-1 | Draft proposal, single target verification, accept/reject, positive residual, bonus token | PARTIAL (`speculative.py::exact_speculative_step`; feeds the prompt twice, trims by input length) |
| S1-2 | Correct cache rollback/advancement | MISSING (`_trim_kv_cache` uses removed `key_cache` attribute) |
| S1-3 | Verify variable proposal lengths, final-token truncation, padding, EOS, one-token and fully accepted rounds | MISSING |
| S1-4 | χ² on a 3-symbol Markov toy, 100,000 length-3 sequences, all expected ≥10, not rejected at p<0.01 | PARTIAL (`SpeculativeSampler.run_chi2_test`: separate code path from the real sampler, γ=1, plain sampling of the next state) |
| S1-5 | Negative control (residual → p) rejected at p<0.01 | PARTIAL (same) |
| S1-6 | Show joint-sequence probabilities, not only marginals | MISSING |
| S1-7 | Real-model split-prefill equivalence and cache-position consistency before timing | MISSING |
| S2-1 | Target-only vs speculative with unchanged draft, γ ∈ {2,4} | MISSING |
| S2-2 | Per prompt: accepted/examined, accepted/proposed, mean accepted prefix per verification, target passes/output token, draft forwards, GPU memory, EOS/truncation flags, paired time and tokens/s | MISSING |
| S2-3 | Warm-setup, prefill and steady-decode measured separately | MISSING |
| S2-4 | Isolated draft/target decode cost at several prompt lengths | MISSING |
| S3-1 | Off-policy run (pool, quotas, steps, loss, LR, lengths, train time, peak memory, teacher-gen cost) | PARTIAL (`distill.py::offpolicy_distill`) |
| S3-2 | No evaluation prefixes in the pool | MISSING (check) |
| S4-1 | On-policy run with same LoRA init, settings, 40k quota; student rollout lengths and sampling cost separate | PARTIAL (`distill.py::onpolicy_distill`) |
| S4-2 | Match math/short quotas; report differences in updates / execution time | MISSING |
| S5-1 | fwd KL, rev KL, TV, greedy agreement, 1−TV on all banks for original/off/on drafts | PARTIAL (`prefix_eval.py::compute_alignment_metrics`) |
| S5-2 | Same prefix IDs/positions, token-weighted denominators | MISSING |
| S5-3 | Split math / familiar / held-out extraction; short vs long positions | MISSING |
| S5-4 | Training-seed dispersion; aligned prefix-level disagreements | MISSING |
| S6-1 | Merge each LoRA into a fresh FP16 0.5B backbone; keep one merged checkpoint per seed/method | MISSING |
| S6-2 | Verify merged vs unmerged logits on the same prompt batch; report error | MISSING |
| S6-3 | Same sampler + EOS-aware 120-prompt eval for original + 4 merged, γ = 2, 4 | MISSING |
| S6-4 | Counterbalanced order; adjacent target-only timing for every condition | MISSING |
| S6-5 | Acceptance, output tokens, verification work, draft work, per-prompt throughput ratios, peak memory | MISSING |
| S6-6 | Per-round synchronized decomposition: draft / verify / accept+residual / cache rollback; prefill and host overhead separate | PARTIAL (`common/timing.py::timed_steps`, unused) |
| S6-7 | c, v_γ, projected Ŝ_γ, measured S_γ, break-even α (or nonexistence) per γ | MISSING |
| S6-8 | One unmerged adapter timed separately (wrapper overhead) | MISSING |
| S7-1 | Outcomes by math / familiar / shifted and by output-length bins | MISSING |
| S7-2 | Expected acceptance on banks vs measured acceptance on trajectories | MISSING |
| S7-3 | Standalone student accuracy at 512/32 caps; math extraction; format compliance | MISSING |
| S7-4 | Deployment break-even including distillation + merge cost | MISSING |
| S-Ev1 | Sampler correctness evidence: χ², negative control, rollback and bonus tests, EOS, target-only reference | PARTIAL |
| S-Ev2 | Paired loss / scored-token curves; model and adapter parameter counts; costs; seed variation | MISSING |
| S-Ev3 | Common-prefix alignment table on 3 shared banks + student banks; target-vs-speculative consistency | MISSING |
| S-Ev4 | Speculative table incl. 4-part decomposition, measured vs predicted, break-even | MISSING |
| S-Ev5 | Paired bootstrap 95% CIs over prompts for throughput and acceptance; seeds separate | PARTIAL (`common/bootstrap.py`) |
| S-Ev6 | EOS/truncation frequency, standalone accuracy, merged vs unmerged timing/error, amortized cost | MISSING |
| S-Ev7 | Representative matched-prompt continuations (if a failure/trade-off is present) | MISSING |

## Repository deliverables (§4)

| ID | Requirement | Status (pre-work) |
|---|---|---|
| R-1 | Code, configs, env spec, data selection scripts, example IDs | PARTIAL |
| R-2 | Machine-readable per-condition / per-example results | Task 1 only |
| R-3 | README with reproduction instructions | PARTIAL (no T2/T3 instructions) |
| R-4 | Attribution for materially reused external code | MISSING (`README.md` "_TODO: attribution._") |
| R-5 | No raw datasets or large checkpoints committed | `.gitignore` |
| R-6 | Cross-task discussion | skipped: optional ("not required", §4) |
| R-7 | Additional readings (§2, §3) | skipped: optional |
