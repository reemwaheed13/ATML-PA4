"""CPU test for scripts/export_results.py: whitelist, deny list, size caps, manifest hashes, zip."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "export_results", os.path.join(os.path.dirname(__file__), "..", "scripts", "export_results.py"))
er = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(er)

MB = 1024 * 1024


def _put(root: Path, rel: str, size: int = 64, byte: bytes = b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((byte * size)[:size])


class TestExport(unittest.TestCase):
    def _run(self, task, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        for rel, size in files.items():
            _put(base / "drive" / f"task{task}", rel, size, byte=rel[:1].encode())
        logs = []
        zip_path = base / "out" / f"export_task{task}.zip"
        man = er.export(base / "drive", task, base / "staging", zip_path, log=logs.append)
        return base, man, logs, zip_path

    def test_task2_whitelist_and_caps(self):
        keep = {
            "tables/b_runs.csv": 100, "tables/aggregate_status.json": 50, "plots/b_curves_x.png": 300,
            "b/grid_manifest.json": 40, "b/runs/r1/summary.json": 30, "b/runs/r1/curves.json": 30,
            "b/forward_sweep/dense_T256.json": 20, "a/conditions/fp16_L256_dense.json": 20,
            "a/checks.json": 20, "c/c1_validation.json": 20, "c/packed_bench.json": 20,
            "c/prompts/p001/summary.json": 20,
        }
        drop = {
            "b/runs/r1/ckpt_step100.pt": 200, "b/_train_cache/train_K8_a_n1.npz": 200,
            "b/runs/r1/optimizer.json": 10, "a/conditions/fp16_L256_dense.error.json": 10,
            "b/runs/r2/error.json": 10, "c/notes.txt": 10, "plots/big.png": 21 * MB,
        }
        base, man, logs, zip_path = self._run(2, {**keep, **drop})
        copied = {f["path"] for f in man["files"]}
        self.assertEqual(copied, set(keep))
        self.assertEqual({s["path"] for s in man["skipped"]}, set(drop))
        reasons = {s["path"]: s["reason"] for s in man["skipped"]}
        self.assertEqual(reasons["plots/big.png"], "file > 20 MB")
        for rel, size in drop.items():   # every skipped file is printed with its size
            self.assertTrue(any(rel in l and f"{size / 1e6:.2f} MB" in l for l in logs), rel)

        dst = base / "staging" / "results" / "task2"
        disk = json.loads((dst / er.MANIFEST).read_text())
        self.assertEqual(disk["files"], man["files"])
        for f in man["files"]:
            data = (dst / f["path"]).read_bytes()
            self.assertEqual(f["size"], len(data))
            self.assertEqual(f["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(data, (base / "drive" / "task2" / f["path"]).read_bytes())
        on_disk = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
        self.assertEqual(on_disk, set(keep) | {er.MANIFEST})

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            self.assertEqual(names, {f"results/task2/{r}" for r in on_disk})
            for f in man["files"]:
                self.assertEqual(hashlib.sha256(zf.read(f"results/task2/{f['path']}")).hexdigest(), f["sha256"])

    def test_task3_weights_and_teacher_pool(self):
        keep = {
            "check.json": 10, "sampler_validation.json": 10, "tables/t3_spec_main.csv": 10,
            "plots/t3_distill_loss.png": 10, "spec/original__g4/primary.jsonl.gz": 10,
            "spec/original__g4/fixed64.jsonl.gz": 10, "spec/original__g4/summary.json": 10,
            "standalone/original/rows.jsonl.gz": 10, "banks/alignment/rows.jsonl.gz": 10,
            "banks/student_off_s6304.jsonl.gz": 10, "distill/off_s6304/curves.json": 10,
            "distill/off_s6304/summary.json": 10, "distill/teacher_pool.meta.json": 10,
            "merge_logs/off_s6304/merge_summary.json": 10, "costs/summary.json": 10,
        }
        drop = {
            "distill/teacher_pool.jsonl.gz": 6 * MB,
            "distill/off_s6304/adapter/adapter_model.safetensors": 100,
            "distill/off_s6304/adapter/adapter_config.json": 10,
            "distill/off_s6304/presentations.jsonl.gz": 10,
            "merged/off_s6304/model.safetensors": 100, "merged/off_s6304/config.json": 10,
            "distill/off_s6304/optimizer.bin": 10, "banks/exact_spec_original_gamma4.jsonl.gz": 21 * MB,
        }
        _, man, _, _ = self._run(3, {**keep, **drop})
        self.assertEqual({f["path"] for f in man["files"]}, set(keep))
        reasons = {s["path"]: s["reason"] for s in man["skipped"]}
        self.assertEqual(set(reasons), set(drop))
        self.assertEqual(reasons["distill/teacher_pool.jsonl.gz"], "teacher-pool rollouts > 5 MB")
        self.assertEqual(reasons["banks/exact_spec_original_gamma4.jsonl.gz"], "file > 20 MB")

    def test_small_teacher_pool_kept_and_rerun_idempotent(self):
        base, man, _, zip_path = self._run(3, {"distill/teacher_pool.jsonl.gz": 4 * MB, "check.json": 10})
        self.assertEqual({f["path"] for f in man["files"]}, {"distill/teacher_pool.jsonl.gz", "check.json"})
        man2 = er.export(base / "drive", 3, base / "staging", zip_path, log=lambda *_: None)
        self.assertEqual(man2["files"], man["files"])

    def test_missing_root(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                er.export(d, 2, Path(d) / "s", None, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
