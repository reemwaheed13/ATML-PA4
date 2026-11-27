"""Regenerate notebooks/02_task2.ipynb and notebooks/03_task3.ipynb from the cell
sources below (style copied from notebooks/01_task1.ipynb: numbered cells, each
printing PASS or FAIL).  Run:  python scripts/make_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

META = {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def write(name, cells):
    nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", name, len(cells), "cells")


# ── shared setup cells ────────────────────────────────────────────────────────

def setup_cells(task, results_sub, files_check):
    return [
        code(f'''
# Cell 1: Drive + HF_HOME + data check
import os, sys, json
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

HF_HOME = '/content/drive/MyDrive/atml_pa4_hf_cache'
DATA_REPO = '/content/drive/MyDrive/atml-pa4-datasets'
DATA_ROOT = DATA_REPO + '/data/public_locked'
RESULTS_ROOT = '/content/drive/MyDrive/atml_pa4_results'
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(RESULTS_ROOT + '/{results_sub}', exist_ok=True)
os.environ['HF_HOME'] = HF_HOME

FILES = {files_check!r}
missing = [f for f in FILES if not os.path.exists(f'{{DATA_REPO}}/{{f}}')]
print('HF_HOME     :', HF_HOME)
print('DATA_REPO   :', DATA_REPO)
print('RESULTS_ROOT:', RESULTS_ROOT + '/{results_sub}')
print('PASS' if not missing else f'FAIL: missing {{missing}}')
'''),
        code('''
# Cell 2: clone / pull repo
import subprocess
REPO_URL = 'https://github.com/reemwaheed13/ATML-PA4'
REPO_DIR = '/content/ATML-PA4'
if os.path.isdir(REPO_DIR + '/.git'):
    r = subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], capture_output=True, text=True)
else:
    r = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
head = subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True).stdout.strip()
print('HEAD:', head)
print('PASS' if r.returncode == 0 and os.path.exists('%s') else 'FAIL')
''' % task),
        code('''
# Cell 3: pinned versions (handout: transformers==4.57.1, huggingface_hub==0.36.2)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'transformers==4.57.1', 'huggingface_hub==0.36.2', 'peft>=0.14.0'], capture_output=True, text=True)
print(r.stderr[-2000:] if r.returncode else 'pip OK')
v = subprocess.run([sys.executable, '-c',
                    'import transformers, huggingface_hub, torch;'
                    'print(transformers.__version__, huggingface_hub.__version__, torch.__version__, torch.cuda.is_available())'],
                   capture_output=True, text=True).stdout.split()
print('transformers, huggingface_hub, torch, cuda:', v)
ok = r.returncode == 0 and v[:2] == ['4.57.1', '0.36.2'] and v[3] == 'True'
print('PASS' if ok else 'FAIL (if transformers was already imported in this kernel, restart the runtime and rerun from cell 1)')
'''),
        code('''
# Cell 4: CPU unit tests
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
print(r.stdout[-4000:])
print(r.stderr[-2000:])
print('PASS' if r.returncode == 0 else 'FAIL')
'''),
    ]


DRIVER_HELPER = '''
def run_driver(module, args):
    cmd = [sys.executable, '-u', '-m', module] + args
    print(' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    return p.wait()
'''

# ── 02_task2 ──────────────────────────────────────────────────────────────────

T2_INTRO = md('''
# 02_task2 — PA4 Task 2 (attention, MQAR, KV memory) on Colab L4

Run top-to-bottom. Every code cell prints `PASS` or `FAIL`. Results go to Drive
(`/content/drive/MyDrive/atml_pa4_results/task2`), never to the repo. Every stage is
resumable: re-running a cell after a disconnect skips finished runs; a failed run writes
`error.json` and the driver continues.

| Cell | What it does | Est. L4 time |
|---|---|---|
| 1 | Mount Drive, set HF_HOME, check the data files | <1 min |
| 2 | Clone / pull the repo | <1 min |
| 3 | Install pinned transformers / huggingface_hub / peft | 1 min |
| 4 | CPU unit tests | 3-5 min |
| 5 | 2B: the 42-run MQAR grid (B1, B2, B3 x 3 seeds) | ~1-1.5 h (first call also builds 6 training streams, ~10 min) |
| 6 | 2B: forward-only sweep T in {256,1024,4096,8192} | ~5 min |
| 7 | 2A: dense / tiled / SDPA sweep (FP32+FP16, L 256..8192) + causality, tile, edge-case checks | ~15-25 min |
| 8 | 2C: C1 cache validation (split prefill, 100% retention, scoring vs SDPA) | ~2 min |
| 9 | 2C: 180 prompts x 6 cache methods (C2-C5) | ~20-30 min |
| 10 | 2C: C5 packed encode/decode benchmark (B 1/8, L 2048/4096) | ~5 min |
| 11 | Aggregate: every Task 2 table and plot | ~2 min |
''')

T2_B_GRID = code('''
# Cell 5: 2B - full 42-run MQAR grid (resumable; rerun after a disconnect)
''' + DRIVER_HELPER + '''
T2 = RESULTS_ROOT + '/task2'
rc = run_driver('task2_attention.run_task2', ['2b', '--stage', 'train', '--runs', 'all',
                '--data-repo', DATA_REPO, '--results-root', T2])
man = json.load(open(f'{T2}/b/grid_manifest.json'))
runs = [g['run'] for g in man['grid']]
done = [r for r in runs if os.path.exists(f'{T2}/b/runs/{r}/summary.json')]
failed = [r for r in runs if os.path.exists(f'{T2}/b/runs/{r}/error.json')]
print(f"data check: {man['data_check']}")
print(f'{len(done)}/{len(runs)} runs complete; failed: {failed}')
for r in done:
    s = json.load(open(f'{T2}/b/runs/{r}/summary.json'))
    c, K = s['test']['cells'], s['K']
    acc = {reg: c['K%d_%s' % (K, reg)]['acc'] for reg in ('near', 'far', 'mixed')}
    flag = '  FAILED-TO-LEARN' if s['failed_to_learn'] else ''
    print('%-30s near %.3f far %.3f mixed %.3f  %.1f ms/upd%s' % (
        r, acc['near'], acc['far'], acc['mixed'], s['train_stats']['ms_per_update'], flag))
print('PASS' if rc == 0 and len(runs) == 42 and len(done) == 42 else 'FAIL')
''')

T2_B_SWEEP = code('''
# Cell 6: 2B - forward-only inference sweep (fixed random embeddings, B=4, H=4, d=32)
rc = run_driver('task2_attention.run_task2', ['2b', '--stage', 'sweep', '--results-root', T2])
import glob
recs = [json.load(open(f)) for f in sorted(glob.glob(f'{T2}/b/forward_sweep/*_T*.json')) if 'error' not in f]
for r in sorted(recs, key=lambda r: (r['method'], r['T'])):
    print(f"{r['method']:17s} T={r['T']:5d} {r['status']:4s} {r.get('mean_ms', float('nan')):9.2f} ms  "
          f"peak {r.get('peak_alloc_mb', float('nan')):8.0f} MB  [{r['materializes']}]")
print('PASS' if rc == 0 and len(recs) == 24 else 'FAIL')
''')


T2_A = code('''
# Cell 7: 2A - attention operators (resumable per dtype x L x method)
rc = run_driver('task2_attention.run_task2', ['2a', '--results-root', T2])
import glob
recs = [json.load(open(f)) for f in sorted(glob.glob(f'{T2}/a/conditions/*.json')) if not f.endswith('.error.json')]
errs = sorted(glob.glob(f'{T2}/a/conditions/*.error.json'))
for r in sorted(recs, key=lambda r: (r['dtype'], r['method'], r['L'])):
    fw = (r.get('forward') or {}).get('mean_ms'); fb = (r.get('forward_backward') or {}).get('mean_ms')
    e = (r.get('output_error') or {}).get('max_abs')
    disp = (r.get('dispatched_forward') or {}).get('backend', '')
    print('%-5s %-15s L=%5d %-11s fwd %9s fwd+bwd %9s max|err| %10s %s' % (
        r['dtype'], r['method'], r['L'], r['status'], '%.2f' % fw if fw else '-', '%.2f' % fb if fb else '-',
        '%.2e' % e if e is not None else '-', disp))
chk = json.load(open(f'{T2}/a/checks.json'))
causal_ok = all(v.get('causal', True) for v in chk['causality'].values())
print('causality:', {k: v.get('causal', v.get('status')) for k, v in chk['causality'].items()})
print('errors:', errs)
print('PASS' if rc == 0 and len(recs) == 6 * 2 * 7 and not errs and causal_ok else 'FAIL')
''')

T2_C1 = code('''
# Cell 8: 2C - C1 validation of the FP16 cache path (run before interpreting compressed caches)
rc = run_driver('task2_attention.run_task2', ['2c', '--stage', 'c1', '--data-repo', DATA_REPO, '--results-root', T2])
c1 = json.load(open(f'{T2}/c/c1_validation.json'))
for c in c1['cases']:
    print(f"{c['id']}: Lctx={c['context_len']} max|dlogit|={c['split_vs_full_max_abs_logit_diff']:.3e} "
          f"argmax agree={c['split_vs_full_argmax_agreement']:.3f} greedy==100%-retention:{c['full_retention_matches_greedy']} "
          f"finite:{c['all_logits_finite']} scoring err={c['scoring_manual_vs_sdpa_max_abs_err']:.2e} "
          f"cache unchanged:{c['scoring_cache_unchanged']}")
print('PASS' if rc == 0 and c1['pass'] else 'FAIL')
''')

T2_C_EVAL = code('''
# Cell 9: 2C - every KV-cache method on all 180 prompts (resumable per prompt)
rc = run_driver('task2_attention.run_task2', ['2c', '--stage', 'eval', '--data-repo', DATA_REPO, '--results-root', T2])
import glob
P = [json.load(open(f)) for f in glob.glob(f'{T2}/c/prompts/*/summary.json')]
errs = glob.glob(f'{T2}/c/prompts/*/error.json')
for m in ['fp16', 'sink_recent', 'question_aware', 'int4_tokenwise', 'int4_channelwise', 'hybrid']:
    R = [p['methods'][m] for p in P]
    if R:
        print('%-17s acc %.3f  frac-of-FP16 %.3f  within budget %s' % (
            m, sum(r['exact_match'] for r in R) / len(R), sum(r['realized_fraction_of_fp16'] for r in R) / len(R),
            all(r['within_budget'] in (None, True) for r in R)))
print(f'{len(P)}/180 prompts, errors: {len(errs)}')
print('PASS' if rc == 0 and len(P) == 180 and not errs else 'FAIL')
''')

T2_C_BENCH = code('''
# Cell 10: 2C - C5 packed-cache encode/decode benchmark (10 warmed, synchronized reps)
rc = run_driver('task2_attention.run_task2', ['2c', '--stage', 'bench', '--data-repo', DATA_REPO, '--results-root', T2])
rows = json.load(open(f'{T2}/c/packed_bench.json'))
for r in rows:
    print('%-32s B=%d L=%d pack %.2f ms unpack %.2f ms step+unpack %.2f ms vs fp16 %.2f ms bytes %d (==formula %s)' % (
        r['layout'], r['B'], r['L'], r['pack_ms']['mean_ms'], r['unpack_dequant_ms']['mean_ms'],
        r['decode_step_with_unpack_ms']['mean_ms'], r['decode_step_fp16_ms']['mean_ms'], r['serialized_bytes'],
        r['bytes_match_formula']))
print('PASS' if rc == 0 and len(rows) == 12 and all(r['bytes_match_formula'] for r in rows) else 'FAIL')
''')

T2_AGG = code('''
# Cell 11: aggregate - every Task 2 table (tables/*.csv|json) and plot (plots/*.png)
rc = run_driver('task2_attention.run_task2', ['aggregate', '--results-root', T2])
st = json.load(open(f'{T2}/tables/aggregate_status.json'))
for part, v in st.items():
    print(part, v.get('status'), {k: v[k] for k in v if k in ('n_conditions', 'n_runs', 'missing_runs', 'n_prompts', 'c1_validation', 'error')})
print('tables:', sorted(os.listdir(f'{T2}/tables')))
print('plots :', sorted(os.listdir(f'{T2}/plots')))
print('PASS' if rc == 0 and all(v.get('status') == 'ok' for v in st.values()) else 'FAIL')
''')


def nb2():
    cells = [T2_INTRO] + setup_cells('task2_attention/run_task2.py', 'task2',
                                     ['data/mqar/mqar_test.jsonl.gz', 'data/mqar/mqar_development.jsonl.gz',
                                      'scripts/mqar_generator.py', 'data/kv_retrieval/kv_retrieval_cases.jsonl.gz'])
    cells += [T2_B_GRID, T2_B_SWEEP, T2_A, T2_C1, T2_C_EVAL, T2_C_BENCH, T2_AGG]
    write("02_task2.ipynb", cells)

# ── 03_task3 ──────────────────────────────────────────────────────────────────

T3_INTRO = md("""
# 03_task3 — PA4 Task 3 (exact speculative sampling + LoRA policy distillation) on Colab L4

Target/teacher `Qwen/Qwen2.5-1.5B-Instruct`, draft/student `Qwen/Qwen2.5-0.5B-Instruct` (FP16, SDPA).
Run top-to-bottom; every code cell prints `PASS` or `FAIL`. Results go to Drive
(`/content/drive/MyDrive/atml_pa4_results/task3`). Every stage is resumable (finished units are
skipped; a failed unit writes `error.json` and the driver continues).

| Cell | Stage | Est. L4 time |
|---|---|---|
| 1-4 | setup + CPU tests | ~6 min |
| 5 | `check`: tokenizer / chat-template agreement, split disjointness, bank prompts | 1 min |
| 6 | `validate`: chi^2 (gamma 2, 4) + negative control, split prefill, cache consistency | 3 min |
| 7 | `bank_spec`: exact_spec_original_gamma4 bank with our sampler | ~8 min |
| 8 | `pool`: frozen teacher rollouts for off-policy | ~5 min |
| 9 | `distill`: off/on-policy x seeds 6304, 6305 | ~40 min |
| 10 | `merge`: 4 merged FP16 checkpoints + merged-vs-unmerged check | ~3 min |
| 11 | `bank_students`: own-policy rollout bank per student | ~3 min |
| 12 | `banks_eval`: KL / TV / acceptance on every bank x 5 drafts | ~10 min |
| 13 | `costs`: isolated draft / target step costs (c, v_gamma) | ~3 min |
| 14 | `spec` original draft (step 2) | ~30 min |
| 15 | `spec` 4 merged drafts (step 6) | ~2 h |
| 16 | `unmerged`: one unmerged adapter, wrapper overhead | ~3 min |
| 17 | `standalone`: each draft's own accuracy (512 / 32 caps) | ~6 min |
| 18 | aggregate | ~2 min |
""")

def t3(cell_no, title, args, check):
    return code(f"""
# Cell {cell_no}: {title}
""" + (DRIVER_HELPER if cell_no == 5 else "") + f"""
T3 = RESULTS_ROOT + '/task3'
rc = run_driver('task3_specdec.run_task3', {args!r} + ['--data-repo', DATA_REPO, '--results-root', T3])
""" + check)

T3_CELLS = [
    t3(5, "check: tokenizer / chat template agreement, train-test disjointness, bank prompt text", ['check'], """
c = json.load(open(f'{T3}/check.json'))
print('template mismatches:', c['chat_template_token_mismatches'], 'vocab equal:', c['vocab_equal'])
print('splits:', c['splits']); print('banks:', c['supplied_banks'])
print(c['example_prompt'])
print('PASS' if rc == 0 and c['pass'] else 'FAIL')
"""),
    t3(6, "validate: chi^2 on the shared round function + real-model cache checks", ['validate'], """
v = json.load(open(f'{T3}/sampler_validation.json'))
for r in v['chi2']:
    print(f"gamma={r['gamma']} correct={r['correct_sampler']}: chi2={r['chi2']:.1f} dof={r['dof']} p={r['p_value']:.3g} "
          f"min expected={r['min_expected']:.0f} rejected={r['rejected_at_0.01']}")
for c in v['real_model_checks']:
    print(c)
ok = v['chi2_pass'] and all(c.get('argmax_equal', True) for c in v['real_model_checks']) and \
     all(c.get('max_verify_logit_err_vs_full_forward', 0) < 0.5 for c in v['real_model_checks'])
print('PASS' if rc == 0 and ok else 'FAIL')
"""),
    t3(7, "bank_spec: exact-speculative consistency bank (original draft, gamma 4, our sampler)", ['bank_spec'], """
m = json.load(open(f'{T3}/banks/exact_spec_original_gamma4.meta.json'))
print('records', m['n_records'], 'seconds', round(m['seconds']))
print('PASS' if rc == 0 and m['n_records'] > 3000 else 'FAIL')
"""),
    t3(8, "pool: frozen teacher rollouts (1 per math prompt, 8 per short prompt)", ['pool'], """
m = json.load(open(f'{T3}/distill/teacher_pool.meta.json'))
print(m)
print('PASS' if rc == 0 and m['n_rollouts'] == 500 + 300 * 8 else 'FAIL')
"""),
    t3(9, "distill: off/on-policy x seeds 6304, 6305 (40,000 teacher-scored tokens each)", ['distill'], """
ok = rc == 0
for n in ['off_s6304', 'off_s6305', 'on_s6304', 'on_s6305']:
    p = f'{T3}/distill/{n}/summary.json'
    if not os.path.exists(p):
        print(n, 'MISSING'); ok = False; continue
    s = json.load(open(p))
    print(n, 'updates', s['updates'], 'skipped', s['skipped_updates'], 'scored', s['scored_tokens'],
          'costs', {k: round(v or 0) for k, v in s['costs_s'].items()})
    ok &= s['scored_tokens'] == {'math': 30000, 'short': 10000} and s['updates'] == 157
print('PASS' if ok else 'FAIL')
"""),
    t3(10, "merge: LoRA -> fresh FP16 backbone (one checkpoint per run), merged vs unmerged logits", ['merge'], """
ok = rc == 0
for n in ['off_s6304', 'off_s6305', 'on_s6304', 'on_s6305']:
    p = f'{T3}/merge_logs/{n}/merge_summary.json'
    if not os.path.exists(p):
        print(n, 'MISSING'); ok = False; continue
    m = json.load(open(p)); print(n, m['verify'], m['seconds'])
    ok &= m['verify']['argmax_agreement'] > 0.99
print('PASS' if ok else 'FAIL')
"""),
    t3(11, "bank_students: each trained student's own frozen rollout bank", ['bank_students'], """
import glob
fs = sorted(glob.glob(f'{T3}/banks/student_*.jsonl.gz')); print(fs)
print('PASS' if rc == 0 and len(fs) == 4 else 'FAIL')
"""),
    t3(12, "banks_eval: alignment of every draft on every bank (same prefix IDs for all drafts)", ['banks_eval'], """
s = json.load(open(f'{T3}/banks/alignment/summary.json'))
print({k: s[k] for k in ('n_records', 'n_rows', 'drafts', 'banks')})
print('PASS' if rc == 0 and s['n_rows'] == 5 * s['n_records'] and len(s['banks']) == 7 else 'FAIL')
"""),
    t3(13, "costs: isolated decode-step / verification latency at cache lengths 64, 256, 1024", ['costs'], """
s = json.load(open(f'{T3}/costs/summary.json'))
for r in s['rows']:
    print(r)
print('PASS' if rc == 0 and len(s['rows']) >= 6 else 'FAIL')
"""),
    t3(14, "spec (step 2): original draft vs target-only, gamma 2 and 4, 120 prompts x 2 reps + fixed-64 diagnostic",
       ['spec', '--drafts', 'original'], """
ok = rc == 0
for g in (2, 4):
    p = f'{T3}/spec/original__g{g}/summary.json'
    ok &= os.path.exists(p)
    if os.path.exists(p):
        print(json.load(open(p)))
print('PASS' if ok else 'FAIL')
"""),
    t3(15, "spec (step 6): the four merged drafts, gamma 2 and 4 (resumable per draft x gamma)",
       ['spec', '--drafts', 'off_s6304,off_s6305,on_s6304,on_s6305'], """
done = [n for n in ['off_s6304', 'off_s6305', 'on_s6304', 'on_s6305'] for g in (2, 4)
        if os.path.exists(f'{T3}/spec/{n}__g{g}/summary.json')]
print(len(done), '/ 8 conditions')
print('PASS' if rc == 0 and len(done) == 8 else 'FAIL')
"""),
    t3(16, "unmerged: one unmerged LoRA adapter (off_s6304, gamma 4, 24 math prompts) - wrapper overhead", ['unmerged'], """
print('PASS' if rc == 0 and os.path.exists(f'{T3}/spec/unmerged_off_s6304__g4/summary.json') else 'FAIL')
"""),
    t3(17, "standalone: each draft as a model (512-token math / 32-token short caps, 2 reps)", ['standalone'], """
import glob
fs = sorted(glob.glob(f'{T3}/standalone/*/summary.json')); print(fs)
print('PASS' if rc == 0 and len(fs) == 5 else 'FAIL')
"""),
    t3(18, "aggregate: every Task 3 table / plot", ['aggregate'], """
st = json.load(open(f'{T3}/tables/aggregate_status.json'))
for k, v in st.items():
    print(k, v if not isinstance(v, dict) or 'traceback' not in v else v['error'])
print('tables:', sorted(os.listdir(f'{T3}/tables')))
print('PASS' if rc == 0 and all(v != 'missing' and not (isinstance(v, dict) and v.get('status') == 'failed')
                                for v in st.values()) else 'FAIL')
"""),
]


def nb3():
    cells = [T3_INTRO] + setup_cells('task3_specdec/run_task3.py', 'task3',
                                     ['data/public_locked/gsm8k_train_locked.jsonl.gz',
                                      'data/public_locked/gsm8k_test_locked.jsonl.gz',
                                      'data/short_form/short_train.jsonl', 'data/short_form/short_test.jsonl',
                                      'data/prefix_banks/prefix_banks_locked.jsonl.gz'])
    cells += T3_CELLS
    write("03_task3.ipynb", cells)


if __name__ == "__main__":
    nb2()
    nb3()
