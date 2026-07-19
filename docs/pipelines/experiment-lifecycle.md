# Experiment Lifecycle

Use this lifecycle for experiments built on the shared infrastructure.

## 1. State the executable question

Record the hypothesis, metric, comparison, and failure modes in the experiment artifact before
running it. Do not encode unimplemented research plans in reference documentation.

## 2. Validate inputs

Parse raw config into frozen dataclasses, assert external schemas, build prompt-disjoint splits,
and verify every fixed response has a raw reward score in the cache.

## 3. Seed and log

Call `seed_all` with the config seed. Create a `RunLogger`: every record goes to a new JSONL file,
with wandb mirroring controlled by `WandbConfig`.

## 4. Build small reusable units

Keep reusable logic in `src/concentration/` and thin orchestration or profiling entry surfaces
outside core modules. Pass masks, spans, and refined config explicitly.

## 5. Test invariants

Add deterministic examples, independent numerical references, and bounded property tests. Use tiny
random models in the default suite and mark real downloads `slow`.

## 6. Record outputs

Keep generated artifacts out of git by default. Check in benchmark baselines only as measured JSON
data, and never overwrite a prior baseline silently.
