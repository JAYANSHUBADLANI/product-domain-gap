"""Grad-CAM on individual shifted-domain misses, illustrative evidence
about what the model appears to be keying on when it is wrong, not a
quantified result. Hooks `mobilenet_v2.conv_1x1`, the last convolutional
feature map before global pooling and the classifier head, the standard
Grad-CAM target: the last spatial layer, so the resulting map is over
the same 7x7 grid the classifier's decision is actually computed from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.mobilenet_v2.conv_1x1.activation
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute(self, pixel_values: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """pixel_values: a single preprocessed image, shape (1, 3, H, W).
        Returns a (7, 7) or model-native-sized CAM in [0, 1] and the
        class the CAM was computed for (the predicted class if
        target_class is None)."""
        self.model.zero_grad()
        # The backbone is frozen for training (requires_grad=False on
        # every conv weight), so with no requires_grad=True anywhere in
        # the input, autograd never builds a graph through it at all and
        # the backward hook below never fires (self.gradients stays
        # None). Marking the input itself as requiring grad is enough:
        # PyTorch marks an operation's output as requiring grad if ANY
        # of its inputs does, independent of whether the operation's own
        # parameters are frozen, so this makes the whole forward pass
        # differentiable with respect to intermediate activations again
        # without touching which weights actually get a gradient step.
        pixel_values = pixel_values.clone().requires_grad_(True)
        logits = self.model(pixel_values=pixel_values).logits
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        logits[0, target_class].backward()

        # global average pool the gradients over the spatial dims to get
        # one importance weight per channel, then weight and sum the
        # activations by it: the Grad-CAM formula.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=False))
        cam = cam[0].detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class


def save_overlay(original_image: Image.Image, cam: np.ndarray, output_path: Path) -> None:
    """Resizes the CAM to the original image size and saves a red-heat
    overlay blended over the original, for visual inspection."""
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    cam_array = np.array(cam_resized).astype(np.float32) / 255.0

    heat = np.zeros((*cam_array.shape, 3), dtype=np.uint8)
    heat[..., 0] = (cam_array * 255).astype(np.uint8)  # red channel carries CAM intensity

    original_array = np.array(original_image.convert("RGB")).astype(np.float32)
    blended = (0.6 * original_array + 0.4 * heat).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)
