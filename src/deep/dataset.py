"""PyTorch Dataset for CamVid binary road segmentation."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
MASK_ROOT = PROJECT_ROOT / "data" / "processed" / "binary_masks"


class CamVidRoadDataset(Dataset):
    """Dataset for binary road segmentation on prepared CamVid masks.

    Images are loaded from data/raw/CamVid/{split}/.
    Masks are loaded from data/processed/binary_masks/{split}/.
    """

    def __init__(
        self,
        split: str,
        image_size: int | tuple[int, int] = 256,
        raw_root: str | Path = RAW_ROOT,
        mask_root: str | Path = MASK_ROOT,
        augment: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.split = split
        self.raw_root = Path(raw_root)
        self.mask_root = Path(mask_root)
        self.augment = augment

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        self.image_dir = self.raw_root / split
        self.mask_dir = self.mask_root / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f"No samples found for split: {split}")

    def _build_samples(self) -> list[tuple[Path, Path]]:
        image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        samples = []
        for image_path in image_paths:
            mask_path = self.mask_dir / image_path.with_suffix(".png").name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            samples.append((image_path, mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = self.image_size[1], self.image_size[0]
        image = image.resize((width, height), resample=Image.BILINEAR)
        mask = mask.resize((width, height), resample=Image.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Light image-only jitter improves robustness without changing labels.
        if random.random() < 0.5:
            brightness = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(brightness)

        if random.random() < 0.5:
            contrast = random.uniform(0.85, 1.15)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self._resize(image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": mask_path.name,
        }


def main() -> None:
    dataset = CamVidRoadDataset(split="train", image_size=256, augment=True)
    sample = dataset[0]
    print(f"Samples: {len(dataset)}")
    print(f"Image tensor shape: {tuple(sample['image'].shape)}")
    print(f"Mask tensor shape: {tuple(sample['mask'].shape)}")
    print(f"Mask values: {torch.unique(sample['mask']).tolist()}")


if __name__ == "__main__":
    main()
