"""Model and preprocessing setup shared by training and evaluation, so
both stages are guaranteed to preprocess images identically.

The locked protocol fixes `supervised_trainable_part: classifier_head`:
the ImageNet backbone stays frozen, only a fresh classifier head sized
to the 20 locked categories is trained. This is transfer learning, not a
novel technique, deliberately: the contribution is the measured gap and
its recovery curve, not the model.
"""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_backbone_and_processor(backbone: str, revision: str):
    model = AutoModelForImageClassification.from_pretrained(backbone, revision=revision)
    processor = AutoImageProcessor.from_pretrained(backbone, revision=revision)
    return model, processor


def build_classifier(backbone: str, revision: str, n_categories: int) -> torch.nn.Module:
    model, _ = load_backbone_and_processor(backbone, revision)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, n_categories)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    return model


def make_transform(processor):
    """Wraps the model's own HF image processor as a torchvision-style
    transform, so training and evaluation preprocess pixels identically
    to how the pretrained backbone expects them, rather than a hand
    rebuilt resize/crop/normalize pipeline that could subtly diverge."""

    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
