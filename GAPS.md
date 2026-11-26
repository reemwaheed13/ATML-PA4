# GAPS — known gaps, decisions, and cross-session notes

Living tracker of scaffold gaps between the code and the PDF spec, plus decisions
made while fixing them. Keep entries short; link code as `path:line`.

Session split (parallel work):
- **Session A** — items 1, 2, 7: perplexity offsets, Wanda/AWQ calibration memory +
  exact masks, gitignore/PDF/GAPS. (Task 1, PDF §1.)
- **Session B** — items 3, 4: `DynamicCache` API and SDPA attention scoring in
  `task2_attention/kv_cache.py`. (PDF §2 Part C.)
- **Session C** — items 5, 6: speculative sampler rewrite + χ² test, distillation
  sampling settings. (PDF §3.)

---

## Session A — resolved

### Item 1 — perplexity offsets (`common/eval_perplexity.py`)
- Reviewed the scored-token convention against PDF §1 "Quality evaluation". The core
  NLL is correct: a scored target at window position `t` is predicted from
  `log_probs[t-1]`, position 0 is context-only and excluded, each token scored once,
  no state carried across windows, and PPL = exp of the **pooled** token NLL.
- Added a guard rejecting `score_target_offsets` outside `[1, L-1]`: offset 0 would
  silently read `log_probs[..., -1]` (negative wrap) and corrupt the pooled NLL
  without raising; `t >= L` runs off the end. No change to correct-path numbers.

### Item 2 — exact masks (shared across P1/P2/P3)
- **Bug:** the per-row prune mask (`scores > threshold` + a tie fix-up loop) was
  duplicated in `prune_magnitude.py` (P1), `prune_wanda.py` (P2), and
  `prune_nonuniform.py` (P3), and **over-pruned** on ties: when the row threshold
  value was shared by more columns than needed (e.g. dead channels scoring 0),
  `scores > threshold` zeroed *all* ties while the fix-up branch (which only ever
  *added* zeros) never fired because its counter was already ≤ 0. This drifts
  achieved sparsity above target and breaks the matched-sparsity P1/P2/P3
  comparison the spec requires ("What to Watch For": P3 eligible-weight sparsity
  must equal P1/P2).
- **Fix:** single shared helper `baseline.exact_row_prune_mask(scores, n_zero)`
  using `topk(largest=False)` + scatter → exactly `n_zero` zeros per row regardless
  of ties. Applied in all three pruning modules.
- **Scope note:** P1/P3 are not strictly "Wanda/AWQ", but the bug is identical and
  the spec mandates one masking convention across P1/P2/P3, so all three were fixed
  via the shared helper (one-line swap each). The 2:4 diagnostic mask in
  `structured_diag.py` uses a different, already-correct top-2-per-group path —
  left unchanged.

### Item 2 — Wanda/AWQ calibration memory
- **Problem:** `prune_wanda.collect_calibration_inputs` stored the full
  `(n_tokens, d_in)` activation matrix for *every* eligible projection. At the PDF §1
  calibration size (128×512 ≈ 65k tokens) this is >100 GB on Qwen2.5-1.5B — a
  guaranteed OOM; the scaffold could not run as written.
- **Fix:** new `baseline.CalibStats` + `baseline.collect_calibration_stats(...)`
  accumulate only the token-axis reductions every consumer actually uses:
  `col_sqnorm` (= Wanda column norm), `col_absmean` (= AWQ act scale / Q3
  sensitivity), optional `gram` (= Σ x xᵀ), and `n_tokens`. All consumers made
  stats-aware with a raw-tensor fallback (`collect_wanda_scores`, `awq_scale_search`,
  `projection_sensitivity`; `structured_diag`/`nonuniform` inherit via
  `collect_wanda_scores`).
- AWQ reconstruction error is now computed exactly from the Gram matrix via
  `‖D Xᵀ‖_F² = tr(D·G·Dᵀ)` — numerically identical to the old X-based form, no full
  activation matrix. Pruning collects with `with_gram=False` (O(d_in)/projection).
- **Remaining cost to watch:** the Gram matrices for AWQ still total ~10 GB CPU RAM
  (dominated by `down_proj`, d_in≈8960 → 321 MB × 28 layers). Fine on a high-RAM /
  A100 Colab; tight on a default 12 GB runtime. If needed later, stream AWQ
  layer-by-layer (collect→quantize→free) to cap peak to one layer (~370 MB).
- `collect_calibration_inputs` is kept (with a memory warning) for tiny debug sets.

### Item 7 — gitignore / PDF / GAPS
- `ATML_PA4.pdf` untracked (`git rm --cached`, kept on disk) and added to
  `.gitignore` — course spec, not part of the public submission repo.
- This file created.

---

## Open / for other sessions

- **Notebook not yet wired to the new calibration API.** `notebooks/00_smoke.ipynb`
  does not call the calibration functions, so nothing breaks, but any real Task-1
  run should use `collect_calibration_stats(...)` (with `with_gram=True` for AWQ/Q3,
  `with_gram=False` for Wanda) instead of `collect_calibration_inputs`.
- ~~`prune_nonuniform.measure_block_sensitivity` is a stub~~ — replaced (P3/Q3 session):
  P3 now uses a block-level Gram sensitivity of the P2 Wanda mask at s_probe = S
  (`prune_nonuniform.block_sensitivity`), a rank-based allocation with exact global zero
  matching (`allocate`) and lambda chosen on the 32 dev sequences (`select_lambda`). The
  OWL weight-only proxy survives as `outlier_proxy_sensitivity()` (rejected alternative,
  recorded). Q3 uses a per-projection Gram error table of the unchanged Q2 quantizer at
  {2,3,4,8} bits (`quant_mixed.bit_error_table`), greedy marginal-efficiency allocation
  under the TOTAL Q2 byte budget (`allocate`) and b_min chosen on dev (`select_bmin`).
- Only `tests/test_proj_enumeration.py` exists; it covers enumeration/tied weights
  and is unaffected by the above changes. No unit tests yet for masks, calibration
  stats, or perplexity offsets.
