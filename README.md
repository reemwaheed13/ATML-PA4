# ATML-PA4

This repository contains the implementation for ATML Programming Assignment 4, covering
(1) model compression via structured pruning and quantization-aware perplexity calibration,
(2) attention KV-cache quantization with group-wise asymmetric quantization and a
question-aware DynamicCache decode path, and (3) speculative decoding with a draft/target
model pair. Code is developed locally and executed on Colab; the Qwen2.5 model family
(1.5B base/instruct target, 0.5B instruct draft) backs the attention and speculative-decoding
experiments.

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

## Attribution

_TODO: attribution._
