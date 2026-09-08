"""Central deterministic seed handling."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any


ROOT_SEED = 20260908


def stable_hash(*parts: Any, seed: int = ROOT_SEED) -> str:
    """Return a stable SHA256 hex digest for a seeded sequence of values."""

    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(namespace: str, root_seed: int = ROOT_SEED) -> int:
    """Derive a reproducible 32 bit seed for one independent operation."""

    return int(stable_hash(namespace, seed=root_seed)[:8], 16)


def seed_everything(seed: int = ROOT_SEED, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch when those libraries are installed."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    return seed
