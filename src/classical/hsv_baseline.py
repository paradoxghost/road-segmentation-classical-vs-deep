"""HSV/color-thresholding baseline for binary road segmentation."""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image

try:
    from .postprocess import postprocess_mask, to_uint8_mask
except ImportError:
    from postprocess import postprocess_mask, to_uint8_mask


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "predictions" / "hsv"


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image as RGB."""
    return np.array(Image.open(image_path).convert("RGB"))


def lower_image_prior(height: int, width: int, start_fraction: float = 0.30) -> np.ndarray:
    """Create a mask favoring the lower image area where roads usually appear."""
    y_coordinates = np.arange(height, dtype=np.float32)[:, None]
    y_normalized = y_coordinates / max(height - 1, 1)
    return np.repeat(y_normalized >= start_fraction, width, axis=1)


def predict_hsv_mask(
    image_rgb: np.ndarray,
    use_lower_prior: bool = True,
    apply_postprocess: bool = True,
) -> np.ndarray:
    """Predict a road mask using simple HSV road-color rules.

    Asphalt is often low-saturation gray/brown with moderate brightness. The
    thresholds below intentionally avoid very bright sky/buildings and very
    saturated objects such as signs, vegetation, or cars.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    gray_asphalt = (saturation <= 75) & (value >= 35) & (value <= 225)
    dark_asphalt = (saturation <= 105) & (value >= 25) & (value <= 150)

    # Brownish asphalt and shadows can have a little more saturation.
    brown_asphalt = (
        ((hue <= 25) | (hue >= 165))
        & (saturation <= 120)
        & (value >= 35)
        & (value <= 190)
    )

    road_candidate = gray_asphalt | dark_asphalt | brown_asphalt

    if use_lower_prior:
        height, width = road_candidate.shape
        road_candidate &= lower_image_prior(height, width, start_fraction=0.30)

    mask = road_candidate.astype(np.uint8) * 255

    if apply_postprocess:
        mask = postprocess_mask(mask, min_area=700)

    return to_uint8_mask(mask)


def list_images(image_dir: Path) -> list[Path]:
    """List input image files."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(image_path: str | Path, output_path: str | Path) -> Path:
    """Generate and save one HSV baseline prediction."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = load_rgb_image(image_path)
    prediction = predict_hsv_mask(image_rgb)

    Image.fromarray(prediction).save(output_path)
    return output_path


def process_split(
    split: str,
    image_root: str | Path = RAW_ROOT,
    output_root: str | Path = PREDICTIONS_ROOT,
) -> list[Path]:
    """Run HSV thresholding on every image in one split."""
    image_dir = Path(image_root) / split
    output_dir = Path(output_root) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for image_path in list_images(image_dir):
        output_path = output_dir / image_path.with_suffix(".png").name
        saved_paths.append(process_image(image_path, output_path))

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HSV/color road-segmentation baseline.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    saved_paths = process_split(args.split)
    print(f"Saved {len(saved_paths)} HSV predictions to {PREDICTIONS_ROOT / args.split}")


if __name__ == "__main__":
    main()
