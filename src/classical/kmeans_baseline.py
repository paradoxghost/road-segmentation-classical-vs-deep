"""K-means clustering baseline for binary road segmentation."""

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
KMEANS_RAW_ROOT = PROJECT_ROOT / "results" / "predictions" / "kmeans_raw"
KMEANS_MORPH_ROOT = PROJECT_ROOT / "results" / "predictions" / "kmeans_morph"
KMEANS_ROOT = PROJECT_ROOT / "results" / "predictions" / "kmeans"


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image as RGB."""
    return np.array(Image.open(image_path).convert("RGB"))


def resize_for_kmeans(image_rgb: np.ndarray, max_side: int = 320) -> tuple[np.ndarray, float]:
    """Resize an image for faster clustering while keeping aspect ratio."""
    height, width = image_rgb.shape[:2]
    largest_side = max(height, width)

    if largest_side <= max_side:
        return image_rgb, 1.0

    scale = max_side / largest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized, scale


def build_pixel_features(image_rgb: np.ndarray) -> np.ndarray:
    """Build RGB, HSV, and coordinate features for each pixel."""
    height, width = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb = image_rgb.astype(np.float32)

    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    y_normalized = y_coords / max(height - 1, 1)
    x_normalized = x_coords / max(width - 1, 1)

    features = np.stack(
        [
            rgb[..., 0] / 255.0,
            rgb[..., 1] / 255.0,
            rgb[..., 2] / 255.0,
            hsv[..., 0] / 179.0,
            hsv[..., 1] / 255.0,
            hsv[..., 2] / 255.0,
            y_normalized,
            x_normalized,
        ],
        axis=-1,
    )

    # Spatial y-position is important because road pixels usually occupy the
    # lower part of driving images. x-position gets a small weight only.
    weights = np.array([0.8, 0.8, 0.8, 0.7, 1.0, 0.8, 1.4, 0.2], dtype=np.float32)
    return (features * weights).reshape(-1, features.shape[-1]).astype(np.float32)


def run_kmeans(features: np.ndarray, k: int = 5, sample_size: int = 25000) -> np.ndarray:
    """Cluster pixels with OpenCV K-means and return a label for every pixel."""
    rng = np.random.default_rng(42)

    if len(features) > sample_size:
        sample_indices = rng.choice(len(features), size=sample_size, replace=False)
        training_features = features[sample_indices]
    else:
        training_features = features

    cv2.setRNGSeed(42)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
    _, _, centers = cv2.kmeans(
        training_features,
        k,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )

    # Assign every pixel to its nearest learned center.
    distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def select_road_cluster(labels_2d: np.ndarray, image_rgb: np.ndarray, k: int) -> int:
    """Select the cluster most likely to be road using color and spatial cues."""
    height, width = labels_2d.shape
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb_float = image_rgb.astype(np.float32) / 255.0

    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    y_normalized = y_coords / max(height - 1, 1)
    x_normalized = x_coords / max(width - 1, 1)

    lower_half = y_normalized >= 0.50
    upper_half = y_normalized < 0.50
    bottom_center = (
        (y_normalized >= 0.70)
        & (x_normalized >= 0.30)
        & (x_normalized <= 0.70)
    )

    scores = []
    for cluster_id in range(k):
        cluster_mask = labels_2d == cluster_id
        if not np.any(cluster_mask):
            scores.append(-np.inf)
            continue

        lower_fraction = float(np.mean(cluster_mask[lower_half]))
        upper_fraction = float(np.mean(cluster_mask[upper_half]))
        bottom_center_fraction = float(np.mean(cluster_mask[bottom_center]))
        mean_y = float(np.mean(y_normalized[cluster_mask]))

        mean_saturation = float(np.mean(hsv[..., 1][cluster_mask]) / 255.0)
        mean_value = float(np.mean(hsv[..., 2][cluster_mask]) / 255.0)
        color_std = float(np.mean(np.std(rgb_float[cluster_mask], axis=1)))

        low_saturation_score = 1.0 - min(mean_saturation, 1.0)
        gray_score = 1.0 - min(color_std / 0.35, 1.0)
        medium_value_score = 1.0 - min(abs(mean_value - 0.45) / 0.45, 1.0)

        score = (
            3.0 * bottom_center_fraction
            + 1.5 * lower_fraction
            - 1.2 * upper_fraction
            + 0.7 * mean_y
            + 0.8 * low_saturation_score
            + 0.5 * gray_score
            + 0.4 * medium_value_score
        )
        scores.append(score)

    return int(np.argmax(scores))


def predict_kmeans_masks(
    image_rgb: np.ndarray,
    k: int = 5,
    max_side: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict raw and morphology-refined K-means road masks."""
    original_height, original_width = image_rgb.shape[:2]
    small_image, _ = resize_for_kmeans(image_rgb, max_side=max_side)
    small_height, small_width = small_image.shape[:2]

    features = build_pixel_features(small_image)
    labels = run_kmeans(features, k=k).reshape(small_height, small_width)
    road_cluster = select_road_cluster(labels, small_image, k=k)

    small_raw_mask = (labels == road_cluster).astype(np.uint8) * 255
    raw_mask = cv2.resize(
        small_raw_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    morph_mask = postprocess_mask(raw_mask, min_area=900)

    return to_uint8_mask(raw_mask), to_uint8_mask(morph_mask)


def list_images(image_dir: Path) -> list[Path]:
    """List input image files."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(
    image_path: str | Path,
    raw_output_path: str | Path,
    morph_output_path: str | Path,
    canonical_output_path: str | Path | None = None,
    k: int = 5,
    max_side: int = 320,
) -> tuple[Path, Path]:
    """Generate and save one K-means raw and morphology-refined prediction."""
    image_path = Path(image_path)
    raw_output_path = Path(raw_output_path)
    morph_output_path = Path(morph_output_path)

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    morph_output_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = load_rgb_image(image_path)
    raw_mask, morph_mask = predict_kmeans_masks(image_rgb, k=k, max_side=max_side)

    Image.fromarray(raw_mask).save(raw_output_path)
    Image.fromarray(morph_mask).save(morph_output_path)

    if canonical_output_path is not None:
        canonical_output_path = Path(canonical_output_path)
        canonical_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(morph_mask).save(canonical_output_path)

    return raw_output_path, morph_output_path


def process_split(
    split: str,
    image_root: str | Path = RAW_ROOT,
    raw_output_root: str | Path = KMEANS_RAW_ROOT,
    morph_output_root: str | Path = KMEANS_MORPH_ROOT,
    canonical_output_root: str | Path | None = KMEANS_ROOT,
    k: int = 5,
    max_side: int = 320,
) -> tuple[list[Path], list[Path]]:
    """Run K-means clustering on every image in one split."""
    image_dir = Path(image_root) / split
    raw_output_dir = Path(raw_output_root) / split
    morph_output_dir = Path(morph_output_root) / split
    canonical_output_dir = Path(canonical_output_root) / split if canonical_output_root else None

    raw_paths = []
    morph_paths = []

    for image_path in list_images(image_dir):
        filename = image_path.with_suffix(".png").name
        raw_output_path = raw_output_dir / filename
        morph_output_path = morph_output_dir / filename
        canonical_output_path = canonical_output_dir / filename if canonical_output_dir else None

        raw_path, morph_path = process_image(
            image_path=image_path,
            raw_output_path=raw_output_path,
            morph_output_path=morph_output_path,
            canonical_output_path=canonical_output_path,
            k=k,
            max_side=max_side,
        )
        raw_paths.append(raw_path)
        morph_paths.append(morph_path)

    return raw_paths, morph_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the K-means road-segmentation baseline.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-side", type=int, default=320)
    args = parser.parse_args()

    raw_paths, morph_paths = process_split(args.split, k=args.k, max_side=args.max_side)
    print(f"Saved {len(raw_paths)} K-means raw predictions to {KMEANS_RAW_ROOT / args.split}")
    print(f"Saved {len(morph_paths)} K-means morph predictions to {KMEANS_MORPH_ROOT / args.split}")


if __name__ == "__main__":
    main()
