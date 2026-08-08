"""Helpers for experiment directory naming."""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Optional


def build_default_experiment_dir(
    *,
    job_prefix: Optional[str] = None,
    domain: Optional[str] = None,
    run_index: int = 1,
    suffix: Optional[int] = None,
    base_root: str = "experiments",
) -> str:
    """Build a default experiment directory name.

    Resolution order for the name prefix:
    1. ``job_prefix`` (e.g. ``pipeline_hf``, ``pipeline_crafter``)
    2. ``pipeline_{domain}`` when *domain* is set (e.g. ``pipeline_crafter``)
    3. ``experiment``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (job_prefix or "").strip()
    if not prefix and domain:
        prefix = f"pipeline_{domain.strip().lower()}"
    if not prefix:
        prefix = "experiment"

    token = suffix if suffix is not None else random.randint(1000, 9999999)
    dirname = f"{prefix}_{timestamp}_run{run_index}_{token}"
    return os.path.join(base_root, dirname)
