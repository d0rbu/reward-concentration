# Architecture

`reward-concentration` uses a `src` package. Core modules never import test or benchmark entry
surfaces.

```text
src/concentration/
├── cli.py
├── benchmark.py
├── config.py
├── data/
│   ├── preference.py
│   └── safety_eval.py
├── eval/
│   ├── capability.py
│   └── stats.py
├── models/
│   ├── policy.py
│   └── reward_model.py
├── run_logging.py
├── seeding.py
├── tracka/
│   └── sft.py
└── types.py
```

## Types and configuration

`types.py` owns scalar refinements and the three tensor smart constructors. Orthonormal, pooled,
and score tensors are finite fp32 values; invalid dtype, rank, emptiness, or semantics raises at
construction.

`config.py` owns frozen configuration dataclasses and string-valued enums. Raw config enters through
`from_raw` methods, which refine scalar values and enum strings before construction. The optional
KL anchor is represented as `KLAnchorConfig | None`; its fields cannot appear independently.

## Preference data

`data/preference.py` loads `PKU-Alignment/PKU-SafeRLHF-single-dimension`. The schema observed and
asserted on 2026-07-19 is identical for `train` (72,996 rows) and `test` (8,109 rows):

| Field | Datasets type |
|---|---|
| `prompt` | `string` |
| `response_0` | `string` |
| `response_1` | `string` |
| `prompt_source` | `string` |
| `response_0_source` | `string` |
| `response_1_source` | `string` |
| `better_response_id` | `int64` constrained to 0 or 1 |
| `response_0_sha256` | `string`, verified as SHA-256 of `prompt + response_0` |
| `response_1_sha256` | `string`, verified as SHA-256 of `prompt + response_1` |

The upstream splits share prompts, so the loader merges them before assigning sorted, seeded unique
prompts to `train`, `heldout_probe_train`, and `heldout_probe_test`. Responses are deduplicated by
`(prompt, response)` while pair records retain `better_response_id`; this preserves relative labels
when the same response occurs in more than one comparison.

The live data contains blank responses. They are counted and excluded from response records, and a
pair containing one is excluded from the retained comparison list. Overlength responses are then
counted and removed after chat templating. A split with no response or no complete comparison raises.

Chat spans are derived from rendered-template lengths and tokenizer offset mappings. The code never
searches for response text inside the rendered conversation. `loss_span` runs from the start of
assistant content to the end of the rendered conversation — the end-of-turn token plus any trailing
template text (`<|im_end|>` and a newline for Qwen3, pinned by the live-template test); `pool_span`
contains assistant content only. Qwen3 rendering always passes `enable_thinking=False`.

## Safety-evaluation data

`data/safety_eval.py` loads the single 700-row `test` split of
`PKU-Alignment/BeaverTails-Evaluation` and asserts:

| Field | Datasets type | Additional constraint |
|---|---|---|
| `prompt` | `string` | non-blank |
| `category` | `string` | non-blank |
| `category_id` | `int64` | integer in `[0, 13]` |

## Safety-SFT

`tracka/sft.py` consumes the complete Phase 1 preference pipeline and never reloads or retokenizes
raw text inside TRL. Only kept pairs in the project `train` split are considered. Each pair selects
its `better_response_id`, and those selections are deduplicated by `(prompt, response)`. Selection
is existential: a response preferred in one comparison remains an SFT item even when another
comparison labels the same key dispreferred. The source corpus has 579 such dual-labeled keys;
deciding how they enter reward-direction fitting is deliberately deferred to Phase 3.

For the default seed-0 splits and Qwen3 tokenizer observed on 2026-07-19, length filtering removes
6 overlong train responses and 6 incomplete train pairs, leaving 65,072 complete pairs and 126,067
unique responses. Preferred selection yields 63,247 SFT items after removing 1,825 repeated
preferred selections; 469 selected train keys are dual-labeled. Held-out prompts cannot enter this
derivation.

Every item carries Phase 1 `input_ids` and labels copied from `input_ids` only inside `loss_span`.
Prompt positions and all padding positions are `-100`; the loss span includes the assistant
end-of-turn token and trailing template text. A custom collator right-pads with the tokenizer's real
pad token ID, emits its attention mask, and pads labels with `-100`.

The installed TRL 1.8.0 path is `SFTTrainer` with a pretokenized `datasets.Dataset` containing
`input_ids` and `labels`, `dataset_kwargs={"skip_prepare_dataset": True}`, no packing, and no TRL
completion/string masking. `seed_all` runs before dataset derivation and trainer construction.

An SFT run refuses a non-empty output directory. The configured directory is the complete
checkpoint:

```text
<output_dir>/
├── config.json and model.safetensors  model `save_pretrained` output
├── tokenizer.json and tokenizer_config.json  tokenizer `save_pretrained` output
├── effective-config.json             fully defaulted project config
└── run-manifest.json                 config echo, data counts, loss history, final metrics, TRL version
```

## Capability smoke metric

`eval/capability.py` computes fp32, response-token-mean causal NLL and its exact exponential from
pre-masked Phase 1 labels. It shifts labels once, excludes `-100`, and performs exactly one policy
forward per input batch. The `ppl` command aggregates token sums across batches over
`heldout_probe_train`; `--count` caps responses for smoke runs.

## Policy extraction

`models/policy.py` loads a Hugging Face causal LM and tokenizer from an exact model ID and revision.
For Qwen3, `forward_at_layer` attaches a pre-forward hook to decoder block `l`. The captured block
input exactly equals `output_hidden_states=True` entry `[l]`; logits and hidden states come from the
same model call. Hidden states are cast to fp32 immediately.

`pool_hidden_states` accepts an explicit mask and supports mean, last selected token, elementwise
maximum, and elementwise minimum. All four modes are insensitive to left versus right padding and
require at least one selected token per batch item.

## Reward scoring and cache

`models/reward_model.py` defines the `RewardModel` protocol and a frozen sequence-classification
adapter. Each prompt-response pair is rendered with the reward tokenizer's own chat template and
thinking disabled. The model must return `[batch, 1]` logits; those logits are cast to fp32 and
returned without standardization or calibration.

Fixed scores are stored as `scores.safetensors` plus `index.json`. A key hashes the four
components injectively — each component is digested to a fixed 32 bytes first, so distinct
`(prompt, response)` pairs can never collide across field boundaries:

```text
sha256(sha256(rm_id) || sha256(chat_template_hash) || sha256(prompt) || sha256(response))
```

Loading validates metadata, tensor dtype/shape/finiteness, exactly one key per tensor row with
unique contiguous indices, and file contents. Verification requires the cache key set to match the requested fixed dataset exactly;
lookup of a missing key raises.

## Shared utilities

- `eval/stats.py`: torch-only mean/95% CI, Welch t, R-squared, and pairwise binary AUC.
- `seeding.py`: Python, torch CPU/CUDA, Transformers, and CUDA deterministic-workspace seeding.
- `run_logging.py`: a new JSONL file for every run plus optional online/offline wandb mirroring.
- `benchmark.py`: synchronized end-to-end timing and non-overwriting JSON result serialization.

## Test isolation

Default tests set deterministic torch algorithms, fixed seeds, and disabled wandb. Offline Hugging
Face environment variables are exported before collection (import-time snapshots make later
per-test setting ineffective), and non-slow tests run with `socket` patched to raise on any
connection attempt. Tiny random Qwen3 causal and sequence-classification models cover the model
codepaths on CPU. Live dataset/tokenizer/model checks are marked `slow` and run separately.
