from pathlib import Path
import json
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "CamVid"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "dataset_examples"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
EXPECTED_MASK_VALUES = {0, 255}
RANDOM_SEED = 42


def relative_path(path: Path) -> str:
    """Return a clean project-relative path for readable terminal output."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_summary() -> dict:
    """Load the dataset summary created by src/prepare_data.py."""
    summary_path = PROCESSED_ROOT / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_class_dict() -> pd.DataFrame:
    """Read CamVid class_dict.csv and check that the Road color is present."""
    class_dict_path = RAW_ROOT / "class_dict.csv"
    if not class_dict_path.exists():
        raise FileNotFoundError(f"class_dict.csv not found: {class_dict_path}")

    class_dict = pd.read_csv(class_dict_path)
    class_dict.columns = [column.strip().lower() for column in class_dict.columns]

    if "name" not in class_dict.columns:
        raise ValueError(f"'name' column not found in {class_dict_path}")

    road_rows = class_dict[class_dict["name"].str.strip().str.lower() == "road"]
    if road_rows.empty:
        raise ValueError("Road class not found in class_dict.csv")

    road_row = road_rows.iloc[0]
    road_rgb = (int(road_row["r"]), int(road_row["g"]), int(road_row["b"]))
    print(f"Road class color: RGB{road_rgb}")

    return class_dict


def count_files(directory: Path) -> list[Path]:
    """Count image-like files in one directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label_path(image_path: Path, label_dir: Path) -> Path:
    candidates = [
        label_dir / image_path.name,
        label_dir / f"{image_path.stem}_L.png",
        label_dir / f"{image_path.stem}_label.png",
        label_dir / f"{image_path.stem}_gt.png",
        label_dir / f"{image_path.stem}_mask.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No label found for {image_path.name}")


def print_summary(summary: dict) -> None:
    """Print a compact summary table from dataset_summary.json."""
    rows = []
    for split_info in summary.get("splits", []):
        rows.append(
            {
                "Split": split_info["split"],
                "Images": split_info["images_found"],
                "Converted Masks": split_info["converted_masks"],
                "Missing Labels": split_info["missing_labels"],
                "Road Pixel Ratio": f"{split_info['road_pixel_ratio']:.2%}",
            }
        )

    print("Dataset summary:")
    print(pd.DataFrame(rows).to_string(index=False))


def verify_dataset_files() -> dict[str, dict[str, object]]:
    """Verify image, label, and binary mask files for every split."""
    split_records = {}

    for split in SPLITS:
        image_dir = RAW_ROOT / split
        label_dir = RAW_ROOT / f"{split}_labels"
        mask_dir = PROCESSED_ROOT / "binary_masks" / split

        images = count_files(image_dir)
        labels = count_files(label_dir)
        masks = count_files(mask_dir)

        missing_labels = []
        missing_masks = []

        for image_path in images:
            try:
                find_label_path(image_path, label_dir)
            except FileNotFoundError:
                missing_labels.append(image_path.name)

            binary_mask_path = mask_dir / image_path.with_suffix(".png").name
            if not binary_mask_path.exists():
                missing_masks.append(binary_mask_path.name)

        if missing_labels:
            raise FileNotFoundError(
                f"{split}: missing original labels for {len(missing_labels)} images. "
                f"First missing label: {missing_labels[0]}"
            )

        if missing_masks:
            raise FileNotFoundError(
                f"{split}: missing binary masks for {len(missing_masks)} images. "
                f"First missing mask: {missing_masks[0]}"
            )

        split_records[split] = {
            "images": images,
            "labels": labels,
            "masks": masks,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "mask_dir": mask_dir,
        }

    return split_records


def print_split_counts(split_records: dict[str, dict[str, object]]) -> None:
    """Print image, original-label, and binary-mask counts for each split."""
    print("\nDataset split counts:")
    for split in SPLITS:
        record = split_records[split]
        print(f"{split} images: {len(record['images'])}")
        print(f"{split} original labels: {len(record['labels'])}")
        print(f"{split} binary masks: {len(record['masks'])}")


def verify_image_and_mask_sizes(split_records: dict[str, dict[str, object]]) -> int:
    """Check that each image, original label, and binary mask share one size."""
    checked_samples = 0

    for split in SPLITS:
        for image_path in split_records[split]["images"]:
            label_path = find_label_path(image_path, split_records[split]["label_dir"])
            binary_mask_path = split_records[split]["mask_dir"] / image_path.with_suffix(".png").name

            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(label_path) as label:
                label_size = label.size

            with Image.open(binary_mask_path) as binary_mask:
                binary_mask_size = binary_mask.size

            if image_size != label_size or image_size != binary_mask_size:
                raise ValueError(
                    f"Size mismatch for {split}/{image_path.name}: "
                    f"image={image_size}, label={label_size}, binary_mask={binary_mask_size}"
                )

            checked_samples += 1

    return checked_samples


def verify_binary_masks(split_records: dict[str, dict[str, object]]) -> int:
    """Check that every binary mask contains only 0 and 255."""
    checked_masks = 0

    for split in SPLITS:
        for mask_path in split_records[split]["masks"]:
            mask_array = np.array(Image.open(mask_path))
            unique_values = set(np.unique(mask_array).astype(int).tolist())

            if not unique_values.issubset(EXPECTED_MASK_VALUES):
                invalid_values = sorted(unique_values - EXPECTED_MASK_VALUES)
                raise ValueError(
                    f"Invalid values in {mask_path}: {invalid_values}. "
                    "Expected only {0, 255}."
                )

            checked_masks += 1

    return checked_masks


def save_dataset_examples(
    split_records: dict[str, dict[str, object]],
    num_examples: int = 10,
    split: str = "train",
) -> list[Path]:
    """Save side-by-side input, original mask, and binary-mask examples."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    images = split_records[split]["images"]
    if len(images) < num_examples:
        raise ValueError(f"Need {num_examples} images, but only found {len(images)} in {split}")

    rng = random.Random(RANDOM_SEED)
    selected_images = rng.sample(images, num_examples)
    saved_paths = []

    for index, image_path in enumerate(selected_images, start=1):
        label_path = find_label_path(image_path, split_records[split]["label_dir"])
        binary_mask_path = split_records[split]["mask_dir"] / image_path.with_suffix(".png").name

        image = Image.open(image_path).convert("RGB")
        original_mask = Image.open(label_path).convert("RGB")
        binary_mask = Image.open(binary_mask_path).convert("L")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        panels = [
            (image, "Input image", None),
            (original_mask, "Original CamVid mask", None),
            (binary_mask, "Binary road mask", "gray"),
        ]

        for ax, (panel_image, title, cmap) in zip(axes, panels):
            if cmap == "gray":
                ax.imshow(panel_image, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(panel_image)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(image_path.name)
        fig.tight_layout()

        output_path = FIGURES_ROOT / f"dataset_example_{index:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    summary = load_summary()
    print_summary(summary)

    print("\nReading class dictionary...")
    read_class_dict()

    split_records = verify_dataset_files()
    print_split_counts(split_records)

    checked_sizes = verify_image_and_mask_sizes(split_records)
    print(f"\nVerified matching image/mask sizes for {checked_sizes} samples.")

    checked_masks = verify_binary_masks(split_records)
    print(f"\nChecked {checked_masks} binary masks.")

    saved_paths = save_dataset_examples(split_records, num_examples=10)
    figures_dir = relative_path(FIGURES_ROOT)

    print("\nDataset exploration complete.")
    print(f"Saved {len(saved_paths)} example figures to {figures_dir}/")
    print("All binary masks contain only values {0, 255}.")


if __name__ == "__main__":
    main()
