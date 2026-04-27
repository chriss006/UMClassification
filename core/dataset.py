# dataset.py
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

import torch
from torch.utils.data import Dataset


class ImageFolderWithPaths(Dataset):
    def __init__(self, root_dir: str, image_processor, image_extensions):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.root_dir}")

        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        if len(self.classes) == 0:
            raise ValueError(f"No class folders found in {self.root_dir}")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        allowed_exts = set(ext.lower() for ext in image_extensions)

        self.samples = []
        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in allowed_exts:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": label,
            "image_path": image_path,
        }


@dataclass
class ImageClassificationCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        image_paths = [f["image_path"] for f in features]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": image_paths,
        }