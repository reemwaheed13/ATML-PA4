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


def nb2():
    cells = [T2_INTRO] + setup_cells('task2_attention/run_task2.py', 'task2',
                                     ['data/mqar/mqar_test.jsonl.gz', 'data/mqar/mqar_development.jsonl.gz',
                                      'scripts/mqar_generator.py'])
    cells += [T2_B_GRID, T2_B_SWEEP]
    write("02_task2.ipynb", cells)


if __name__ == "__main__":
    nb2()
