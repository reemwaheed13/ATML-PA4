# ATML-PA4

This repository contains the implementation for ATML Programming Assignment 4 on
Qwen2.5 models. Task 1 (model compression, Qwen2.5-1.5B base) covers unstructured
magnitude, Wanda, and nonuniform pruning, a 2:4 structured-sparsity study, and
uniform, AWQ-style, and mixed-precision weight-only post-training quantization.
Task 2 covers exact attention, MQAR (multi-query associative recall), and KV-cache
compression. Task 3 covers speculative decoding with off-policy and on-policy LoRA
draft-model distillation. Code is developed locally and executed on Colab.

## Data

Data comes from [github.com/AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets)
at commit `6a0a9ec`, prepared via `setup_data.py --all`. The pinned upstream source
repositories and revisions used to regenerate the banks are recorded in
`data/local_source_revisions.json`.

## Deviations

(a) The staff `setup_data.py` passed `generator=` to `model.generate()`, which transformers
4.57.1 rejects (`ValueError: model_kwargs not used: ['generator']`). It was replaced with
`torch.manual_seed(<same per-prompt seed>)` before each `generate()` call.

(b) It left Qwen's `generation_config` `repetition_penalty=1.1` in effect. This was set to
`1.0` so the teacher/original-draft banks are pure temperature-1, top-p-1 sampling, as the
handout specifies. Patch: `docs/setup_data_patch.diff`.

(c) WikiText-2 raw rows are lines/paragraphs treated as separate documents: 437 windows
(~115 scored tokens each) for 50k tokens, versus 177 windows for C4. PPL is therefore not
comparable to published full-context WikiText-2 numbers.

(d) AWQ (Q2, and the AWQ-style quantizer reused by Q3) stores serialized FP16
compensating scales instead of folding the inverse scales into a neighbouring layer,
which is the alternative the handout permits. Each projection quantizes W·diag(α) and
keeps α in FP16, costing 2·d_in bytes per projection. Those bytes are counted in the
packed storage of Q2 and in the Q3 budgets, so Q3 is compared with Q2 like-for-like.
The folding in the original scaffold was removed: it applied the compensation twice,
folded through SiLU, scaled the wrong axis of the predecessor, and ignored GQA.

## Attribution

_TODO: attribution._
