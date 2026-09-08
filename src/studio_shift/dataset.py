"""A manifest-driven image dataset. Reads the JSON manifests splits.py
writes (a list of {"path", "category"} rows) rather than scanning a
directory tree at load time, so what a run trains or evaluates on is
exactly what was locked when the manifest was generated, not whatever
happens to be on disk under a folder at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# Manifest rows store paths relative to the repo root (see splits.py), so a
# clone works on any machine. Resolve here rather than at write time.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, categories: list[str], transform=None):
        with open(manifest_path) as f:
            self.rows = json.load(f)
        self.category_to_index = {c: i for i, c in enumerate(categories)}
        self.transform = transform

        unknown = {r["category"] for r in self.rows} - set(self.category_to_index)
        if unknown:
            raise ValueError(f"manifest contains categories not in the locked category list: {unknown}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(self._resolve(row["path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.category_to_index[row["category"]]
        return image, label

    @property
    def categories(self) -> list[str]:
        return list(self.category_to_index)
