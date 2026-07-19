"""Reproducible project-wide random seeding."""

from __future__ import annotations

import os
import random

import torch as t
from beartype import beartype
from transformers import set_seed

from concentration.types import Seed


@beartype
def seed_all(seed: Seed) -> None:
    """Seed Python, torch CPU/CUDA, and Transformers from one refined seed."""
    numeric_seed = int(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(numeric_seed)
    t.manual_seed(numeric_seed)
    t.cuda.manual_seed_all(numeric_seed)
    set_seed(numeric_seed)
