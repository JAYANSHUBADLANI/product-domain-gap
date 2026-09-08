"""Trains the classifier head on the locked product_train split, model
selection on product_val, and locks the resulting checkpoint's own
sha256 so later stages can assert they are evaluating the exact weights
this run produced, not a checkpoint that got silently overwritten since.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import ManifestImageDataset
from .model import build_classifier, load_backbone_and_processor, make_transform, pick_device
from .seed import seed_everything

ROOT = Path(__file__).resolve().parents[2]


def load_protocol() -> dict:
    with open(ROOT / "config" / "protocol.yaml") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(pixel_values=images).logits
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_clean_classifier(
    manifests_dir: Path, checkpoint_path: Path, protocol: dict | None = None,
    epochs: int = 20, batch_size: int = 16, lr: float = 1e-3, patience: int = 5,
) -> dict:
    protocol = protocol or load_protocol()
    root_seed = protocol["project"]["root_seed"]
    categories = protocol["categories"]
    model_cfg = protocol["model"]

    seed_everything(root_seed)
    device = pick_device()

    _, processor = load_backbone_and_processor(model_cfg["backbone"], model_cfg["revision"])
    transform = make_transform(processor)

    train_ds = ManifestImageDataset(manifests_dir / "product_train.json", categories, transform)
    val_ds = ManifestImageDataset(manifests_dir / "product_val.json", categories, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_classifier(model_cfg["backbone"], model_cfg["revision"], len(categories)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    training_seconds = time.time() - t0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

    lock = {
        "checkpoint_path": str(Path(checkpoint_path).resolve().relative_to(ROOT)),
        "sha256": checkpoint_sha256,
        "best_val_acc": best_val_acc,
        "epochs_run": len(history),
        "training_seconds": training_seconds,
        "device": str(device),
        "root_seed": root_seed,
    }
    with open(checkpoint_path.with_suffix(".lock.json"), "w") as f:
        json.dump({"lock": lock, "history": history}, f, indent=2)

    return lock
