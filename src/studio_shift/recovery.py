"""How much of the clean-to-shifted accuracy gap does a small amount of
labelled Real Life data recover, per labelled example added.

Every budget restarts from the same locked clean checkpoint rather than
continuing from the previous budget's fine-tuned weights, so budgets are
independent measurements, not one long training run sliced at
checkpoints. Sampling is nested within each run index: run r's sample
for budget N is a strict prefix of run r's sample for every larger
budget, built from one seeded permutation of the adaptation pool per
category per run, so moving from one budget to the next never swaps out
an already-seen image, only adds new ones. That keeps the recovery
curve's shape attributable to added data, not to which images happened
to be resampled.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import ManifestImageDataset
from .evaluate import bootstrap_class_balanced_ci, class_balanced_accuracy, predict_all
from .model import build_classifier, load_backbone_and_processor, make_transform, pick_device
from .seed import derive_seed


class InMemoryLabeledImages(Dataset):
    def __init__(self, rows: list[dict], categories: list[str], transform):
        self.rows = rows
        self.category_to_index = {c: i for i, c in enumerate(categories)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), self.category_to_index[row["category"]]


def nested_samples_per_run(
    pool_manifest: list[dict], categories: list[str], budgets: list[int], run_index: int, root_seed: int,
) -> dict[int, list[dict]]:
    """budget -> the rows sampled for this run at that budget, nested."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in pool_manifest:
        by_category[row["category"]].append(row)

    max_budget = max(budgets)
    ordered_per_category: dict[str, list[dict]] = {}
    for category in categories:
        rows = list(by_category[category])
        rng = random.Random(derive_seed(f"recovery::run{run_index}::{category}", root_seed))
        rng.shuffle(rows)
        if len(rows) < max_budget:
            raise ValueError(
                f"adaptation pool for {category!r} has only {len(rows)} images, "
                f"fewer than the largest recovery budget {max_budget}"
            )
        ordered_per_category[category] = rows

    result = {}
    for budget in budgets:
        sample = []
        for category in categories:
            sample.extend(ordered_per_category[category][:budget])
        result[budget] = sample
    return result


def fine_tune_from_checkpoint(
    checkpoint_path: Path, backbone: str, revision: str, categories: list[str],
    adaptation_rows: list[dict], device, epochs: int = 10, lr: float = 1e-3, batch_size: int = 8,
) -> torch.nn.Module:
    model = build_classifier(backbone, revision, len(categories))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    if not adaptation_rows:
        model.eval()
        return model

    _, processor = load_backbone_and_processor(backbone, revision)
    transform = make_transform(processor)
    ds = InMemoryLabeledImages(adaptation_rows, categories, transform)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def load_completed_cycles(progress_path: Path) -> dict[tuple[int, int], dict]:
    """Reads a progress .jsonl file (one completed cycle's result per
    line) into a (run, labels_per_class) -> result map, or an empty map
    if the file does not exist yet. A cycle already present here is
    skipped on the next call to run_recovery_curve against the same
    progress_path, so a killed and restarted run resumes rather than
    redoing every cycle from the start.
    """
    if not progress_path.exists():
        return {}
    completed = {}
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[(row["run"], row["labels_per_class"])] = row
    return completed


def run_recovery_curve(
    manifests_dir: Path, checkpoint_path: Path, protocol: dict, progress_path: Path | None = None,
) -> list[dict]:
    """Runs every (run, labels_per_class) cycle, writing each one's
    result to progress_path immediately after it completes (append mode,
    flushed to disk), rather than holding everything in memory until the
    whole sweep finishes. A cycle already recorded in progress_path is
    skipped rather than recomputed, so killing this function partway
    through and calling it again with the same progress_path resumes
    from the next uncompleted cycle instead of starting over.
    """
    categories = protocol["categories"]
    model_cfg = protocol["model"]
    root_seed = protocol["project"]["root_seed"]
    budgets = protocol["recovery"]["labels_per_class"]
    n_runs = protocol["recovery"]["nested_sampling_runs"]

    with open(manifests_dir / "adaptation_pool.json") as f:
        pool_manifest = json.load(f)

    device = pick_device()
    _, processor = load_backbone_and_processor(model_cfg["backbone"], model_cfg["revision"])
    transform = make_transform(processor)
    shifted_ds = ManifestImageDataset(manifests_dir / "shifted_test.json", categories, transform)
    shifted_loader = DataLoader(shifted_ds, batch_size=16, shuffle=False)

    completed = load_completed_cycles(progress_path) if progress_path else {}
    results = list(completed.values())

    total_cycles = n_runs * len(budgets)
    done_count = len(completed)

    for run_index in range(n_runs):
        samples = nested_samples_per_run(pool_manifest, categories, budgets, run_index, root_seed)
        for budget in budgets:
            if (run_index, budget) in completed:
                continue

            model = fine_tune_from_checkpoint(
                checkpoint_path, model_cfg["backbone"], model_cfg["revision"], categories,
                samples[budget], device,
            )
            predictions = predict_all(model, shifted_loader, categories, device)
            acc = class_balanced_accuracy(predictions)
            row = {"run": run_index, "labels_per_class": budget, "class_balanced_accuracy": acc}
            results.append(row)
            done_count += 1

            if progress_path:
                with open(progress_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            print(f"[recovery] cycle {done_count}/{total_cycles} done: "
                  f"run={run_index} labels_per_class={budget} acc={acc:.3f}", flush=True)

    return results


def summarize_recovery_curve(results: list[dict]) -> list[dict]:
    by_budget: dict[int, list[float]] = defaultdict(list)
    for r in results:
        by_budget[r["labels_per_class"]].append(r["class_balanced_accuracy"])
    summary = []
    for budget in sorted(by_budget):
        values = by_budget[budget]
        summary.append({
            "labels_per_class": budget,
            "mean_accuracy": sum(values) / len(values),
            "min_accuracy": min(values),
            "max_accuracy": max(values),
            "n_runs": len(values),
        })
    return summary
