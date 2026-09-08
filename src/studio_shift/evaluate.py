"""Scoring: class-balanced top-1 accuracy (the average of each category's
own accuracy, not overall micro accuracy, so a category with fewer
correct predictions cannot be masked by a larger category doing well),
a bootstrap confidence interval on that number, and a confidence
analysis of shifted-domain misses versus hits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import ManifestImageDataset
from .model import build_classifier, load_backbone_and_processor, make_transform, pick_device
from .seed import derive_seed


@dataclass(frozen=True)
class Prediction:
    true_category: str
    predicted_category: str
    confidence: float
    correct: bool


def load_locked_model(checkpoint_path: Path, backbone: str, revision: str, categories: list[str], device):
    model = build_classifier(backbone, revision, len(categories))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_all(model, loader: DataLoader, categories: list[str], device) -> list[Prediction]:
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            probs = F.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)
            for true_idx, pred_idx, conf in zip(labels.tolist(), preds.cpu().tolist(), confidences.cpu().tolist()):
                predictions.append(Prediction(
                    true_category=categories[true_idx],
                    predicted_category=categories[pred_idx],
                    confidence=conf,
                    correct=true_idx == pred_idx,
                ))
    return predictions


def class_balanced_accuracy(predictions: list[Prediction]) -> float:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for p in predictions:
        by_category[p.true_category].append(p.correct)
    per_category_acc = [sum(v) / len(v) for v in by_category.values()]
    return sum(per_category_acc) / len(per_category_acc)


def per_category_accuracy(predictions: list[Prediction]) -> dict[str, float]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    for p in predictions:
        by_category[p.true_category].append(p.correct)
    return {cat: sum(v) / len(v) for cat, v in by_category.items()}


def bootstrap_class_balanced_ci(
    predictions: list[Prediction], n_boot: int, root_seed: int, namespace: str, alpha: float = 0.05,
) -> dict:
    """Resamples within each category (with replacement) so the
    resample respects the same category-balance the point estimate
    does, then reports a percentile interval over n_boot resamples."""
    by_category: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        by_category[p.true_category].append(p)

    rng = random.Random(derive_seed(namespace, root_seed))
    boot_estimates = []
    for _ in range(n_boot):
        resampled = []
        for cat, preds in by_category.items():
            resampled.extend(rng.choices(preds, k=len(preds)))
        boot_estimates.append(class_balanced_accuracy(resampled))

    boot_estimates.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {
        "point_estimate": class_balanced_accuracy(predictions),
        "ci_low": boot_estimates[lo_idx],
        "ci_high": boot_estimates[hi_idx],
        "n_boot": n_boot,
        "alpha": alpha,
    }


def confidence_gap_analysis(predictions: list[Prediction]) -> dict:
    correct_conf = [p.confidence for p in predictions if p.correct]
    incorrect_conf = [p.confidence for p in predictions if not p.correct]
    return {
        "n_correct": len(correct_conf),
        "n_incorrect": len(incorrect_conf),
        "mean_confidence_correct": sum(correct_conf) / len(correct_conf) if correct_conf else None,
        "mean_confidence_incorrect": sum(incorrect_conf) / len(incorrect_conf) if incorrect_conf else None,
    }


def run_eval(
    manifest_path: Path, checkpoint_path: Path, categories: list[str], backbone: str, revision: str,
    batch_size: int = 16,
) -> list[Prediction]:
    device = pick_device()
    _, processor = load_backbone_and_processor(backbone, revision)
    transform = make_transform(processor)

    dataset = ManifestImageDataset(manifest_path, categories, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_locked_model(checkpoint_path, backbone, revision, categories, device)
    return predict_all(model, loader, categories, device)
