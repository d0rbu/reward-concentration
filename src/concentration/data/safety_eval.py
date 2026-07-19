"""BeaverTails-Evaluation prompt loading with a hard schema boundary."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

from concentration.config import SAFETY_EVAL_DATASET_ID

EXPECTED_SAFETY_FEATURES = Features(
    {
        "prompt": Value("string"),
        "category": Value("string"),
        "category_id": Value("int64"),
    }
)
EXPECTED_SAFETY_ROWS = 700


@dataclass(frozen=True, slots=True)
class SafetyEvalPrompt:
    """One BeaverTails prompt and category label."""

    prompt: str
    category: str
    category_id: int


def assert_safety_schema(dataset: Dataset) -> None:
    """Crash unless a safety-evaluation split exactly matches the observed schema."""
    if dataset.features != EXPECTED_SAFETY_FEATURES:
        raise ValueError(
            f"safety-eval schema changed: expected {EXPECTED_SAFETY_FEATURES}, "
            f"got {dataset.features}"
        )
    if dataset.column_names != list(EXPECTED_SAFETY_FEATURES):
        raise ValueError("safety-eval column order changed")


def extract_safety_prompts(dataset: Dataset) -> tuple[SafetyEvalPrompt, ...]:
    """Validate and materialize safety-evaluation prompt records."""
    assert_safety_schema(dataset)
    prompts: list[SafetyEvalPrompt] = []
    for row in dataset:
        prompt = row["prompt"]
        category = row["category"]
        category_id = int(row["category_id"])
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("safety-eval prompts must be non-blank strings")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("safety-eval categories must be non-blank strings")
        if not 0 <= category_id <= 13:
            raise ValueError("safety-eval category_id must be in [0, 13]")
        prompts.append(SafetyEvalPrompt(prompt, category, category_id))
    if not prompts:
        raise ValueError("safety-eval dataset must be non-empty")
    return tuple(prompts)


def load_safety_eval_prompts(
    dataset_id: str = SAFETY_EVAL_DATASET_ID,
) -> tuple[SafetyEvalPrompt, ...]:
    """Load the real BeaverTails evaluation split and assert schema and row count."""
    loaded = load_dataset(dataset_id)
    if not isinstance(loaded, DatasetDict):
        raise TypeError("safety-eval loader must return a DatasetDict")
    if set(loaded) != {"test"}:
        raise ValueError("safety-eval dataset must contain exactly the test split")
    if len(loaded["test"]) != EXPECTED_SAFETY_ROWS:
        raise ValueError(
            f"safety-eval row count changed: expected {EXPECTED_SAFETY_ROWS}, "
            f"got {len(loaded['test'])}"
        )
    return extract_safety_prompts(loaded["test"])
