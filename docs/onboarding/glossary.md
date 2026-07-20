# Glossary

| Term | Meaning |
|---|---|
| Phantom type | A primitive narrowed by a predicate and represented as a richer static type after validation. |
| Refinement function | A `parse_*` function that validates raw boundary input and returns a phantom type. |
| Runtime boundary | A point where external values enter the package, including config, datasets, tokenizers, model outputs, and cache files. |
| Array contract | A tensor shape and dtype expressed with `jaxtyping` and checked by `beartype`. |
| Smart constructor | A constructor that checks semantic tensor invariants before creating a wrapper. |
| Rank | A strictly positive integer used for counts and later low-rank dimensions. |
| Raw RM score | The unstandardized scalar logit emitted by the configured reward model. |
| Prompt-level split | A partition that assigns every record sharing a prompt to exactly one project split. |
| Loss span | Assistant response tokens through the end of the rendered conversation: the end-of-turn token plus any trailing template text (`<|im_end|>` and a newline for Qwen3). |
| Pool span | Assistant content tokens excluding trailing chat-template special tokens. |
| Score cache | A safetensors score vector plus JSON key-to-row index bound to an RM ID and chat-template hash. |
| Pooling | Mean, last-content-token, elementwise maximum, or elementwise minimum over a mask-selected response span. |
| DANN schedule | The sigmoid gradient-reversal weight ramp from domain-adversarial training (Ganin et al.), one of the configurable lambda-schedule shapes. |
| SFT item | One unique train-split response preferred in at least one kept pair, with its tokenized conversation and loss-span labels. |
| Dual-labeled key | A (prompt, response) key labeled preferred in one comparison and dispreferred in another. |
| Run manifest | The JSON record an SFT run writes beside its checkpoint: schema version, config echo, dataset counts, loss history, final metrics, and the TRL version. |
| Property test | A Hypothesis test that checks an invariant across generated examples. |
| Slow test | A separately selected live-network or GPU integration test excluded from the default suite. |
