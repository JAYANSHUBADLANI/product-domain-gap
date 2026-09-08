"""Deterministic split manifest generation.

Splits are computed from content, not filesystem order: every image's
category is grouped, duplicate groups (identical sha256, see extract.py)
are treated as one atomic unit so an exact duplicate can never land on
both sides of a held-out boundary, and the random assignment within each
category is seeded from the protocol's root seed so a rerun reproduces
the identical manifest byte for byte.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .extract import InventoryEntry
from .seed import derive_seed

ROOT = Path(__file__).resolve().parents[2]


def _group_into_atomic_units(entries: list[InventoryEntry]) -> list[list[InventoryEntry]]:
    """Groups entries sharing a sha256 into one unit; a unique image is
    its own unit of size 1. Units, not individual files, are what gets
    assigned to a split, so an exact duplicate never crosses a boundary.
    """
    by_hash: dict[str, list[InventoryEntry]] = defaultdict(list)
    for e in entries:
        by_hash[e.sha256].append(e)
    return list(by_hash.values())


def build_product_split(
    entries: list[InventoryEntry], category: str, n_train: int, n_val: int, n_test: int,
    root_seed: int,
) -> dict[str, list[InventoryEntry]]:
    """Splits one category's product_images entries into train/val/test.

    Assigns whole atomic (duplicate) units to a split; if a unit would
    straddle a boundary, it is placed entirely in the split it was
    reached at during the shuffle, so real counts can differ slightly
    from n_train/n_val/n_test when a category has any exact duplicates
    among its images. That is reported, not hidden - see
    split_report() below.
    """
    cat_entries = [e for e in entries if e.category == category]
    units = _group_into_atomic_units(cat_entries)

    rng = random.Random(derive_seed(f"product_split::{category}", root_seed))
    rng.shuffle(units)

    train, val, test = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": n_train, "val": n_val, "test": n_test}
    order = ["train", "val", "test"]

    for unit in units:
        # place the whole unit in the first split (in fixed priority
        # order) that still has room, so duplicates never split
        placed = False
        for split_name in order:
            if counts[split_name] + len(unit) <= targets[split_name] or split_name == order[-1]:
                (train if split_name == "train" else val if split_name == "val" else test).extend(unit)
                counts[split_name] += len(unit)
                placed = True
                break
        assert placed

    return {"train": train, "val": val, "test": test}


def build_real_life_split(
    entries: list[InventoryEntry], category: str, n_pool: int, n_shifted_test: int,
    root_seed: int,
) -> dict[str, list[InventoryEntry]]:
    cat_entries = [e for e in entries if e.category == category]
    units = _group_into_atomic_units(cat_entries)

    rng = random.Random(derive_seed(f"real_life_split::{category}", root_seed))
    rng.shuffle(units)

    pool, shifted_test = [], []
    counts = {"pool": 0, "shifted_test": 0}
    targets = {"pool": n_pool, "shifted_test": n_shifted_test}
    order = ["pool", "shifted_test"]

    for unit in units:
        for split_name in order:
            if counts[split_name] + len(unit) <= targets[split_name] or split_name == order[-1]:
                (pool if split_name == "pool" else shifted_test).extend(unit)
                counts[split_name] += len(unit)
                break

    return {"adaptation_pool": pool, "shifted_test": shifted_test}


def build_all_splits(entries: list[InventoryEntry], protocol: dict) -> dict:
    root_seed = protocol["project"]["root_seed"]
    categories = protocol["categories"]
    product_entries = [e for e in entries if e.domain == "product_images"]
    real_life_entries = [e for e in entries if e.domain == "real_life"]

    splits_cfg = protocol["splits"]
    result = {"product_images": {}, "real_life": {}}
    for category in categories:
        result["product_images"][category] = build_product_split(
            product_entries, category,
            splits_cfg["product_images"]["product_train"],
            splits_cfg["product_images"]["product_val"],
            splits_cfg["product_images"]["product_test"],
            root_seed,
        )
        result["real_life"][category] = build_real_life_split(
            real_life_entries, category,
            splits_cfg["real_life"]["adaptation_pool"],
            splits_cfg["real_life"]["shifted_test"],
            root_seed,
        )
    return result


def split_report(splits: dict, protocol: dict) -> list[dict]:
    """Actual vs target counts per category per split, so any duplicate-
    driven deviation from the locked target counts is visible in a
    committed table rather than silently absorbed."""
    rows = []
    splits_cfg = protocol["splits"]
    for category in protocol["categories"]:
        p = splits["product_images"][category]
        rl = splits["real_life"][category]
        rows.append({
            "category": category,
            "product_train": len(p["train"]), "product_train_target": splits_cfg["product_images"]["product_train"],
            "product_val": len(p["val"]), "product_val_target": splits_cfg["product_images"]["product_val"],
            "product_test": len(p["test"]), "product_test_target": splits_cfg["product_images"]["product_test"],
            "adaptation_pool": len(rl["adaptation_pool"]), "adaptation_pool_target": splits_cfg["real_life"]["adaptation_pool"],
            "shifted_test": len(rl["shifted_test"]), "shifted_test_target": splits_cfg["real_life"]["shifted_test"],
        })
    return rows


def _entries_to_manifest(entries: list[InventoryEntry], output_root: Path) -> list[str]:
    # Relative to the repo root, so a manifest is portable across machines
    # and clones instead of baking in the path it happened to be written on.
    return [
        str((output_root / e.domain / e.category / e.filename).relative_to(ROOT))
        for e in entries
    ]


def write_manifests(splits: dict, output_root: Path, manifests_dir: Path) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    flat = {
        "product_train": [], "product_val": [], "product_test": [],
        "adaptation_pool": [], "shifted_test": [],
    }
    for category, s in splits["product_images"].items():
        flat["product_train"] += [
            {"path": p, "category": category} for p in _entries_to_manifest(s["train"], output_root)
        ]
        flat["product_val"] += [
            {"path": p, "category": category} for p in _entries_to_manifest(s["val"], output_root)
        ]
        flat["product_test"] += [
            {"path": p, "category": category} for p in _entries_to_manifest(s["test"], output_root)
        ]
    for category, s in splits["real_life"].items():
        flat["adaptation_pool"] += [
            {"path": p, "category": category} for p in _entries_to_manifest(s["adaptation_pool"], output_root)
        ]
        flat["shifted_test"] += [
            {"path": p, "category": category} for p in _entries_to_manifest(s["shifted_test"], output_root)
        ]

    for name, rows in flat.items():
        with open(manifests_dir / f"{name}.json", "w") as f:
            json.dump(rows, f, indent=2)
