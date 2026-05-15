from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image


def find_road_color(class_dict_path: Path, road_class_name: str = "Road") -> tuple[int, int, int]:
    """
    Read CamVid class_dict.csv and return the RGB color used for the road class.
    Expected columns are usually: name, r, g, b
    """
    df = pd.read_csv(class_dict_path)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    if "name" not in df.columns:
        raise ValueError(f"'name' column not found in {class_dict_path}. Found columns: {df.columns.tolist()}")

    # Try to find the road class
    road_rows = df[df["name"].str.lower().str.strip() == road_class_name.lower()]

    if road_rows.empty:
        print("\nAvailable classes:")
        print(df["name"].tolist())
        raise ValueError(f"Could not find class '{road_class_name}' in class_dict.csv")

    row = road_rows.iloc[0]

    # Accept either r/g/b or red/green/blue column names
    r_col = "r" if "r" in df.columns else "red"
    g_col = "g" if "g" in df.columns else "green"
    b_col = "b" if "b" in df.columns else "blue"

    road_color = (int(row[r_col]), int(row[g_col]), int(row[b_col]))
    return road_color


def convert_mask_to_binary(mask_path: Path, road_color: tuple[int, int, int]) -> np.ndarray:
    """
    Convert a CamVid RGB mask into a binary mask:
    road pixels = 255
    non-road pixels = 0
    """
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)

    road_color_np = np.array(road_color, dtype=np.uint8)

    road_pixels = np.all(mask_np == road_color_np, axis=-1)
    binary_mask = road_pixels.astype(np.uint8) * 255

    return binary_mask


def count_road_pixels(binary_mask: np.ndarray) -> tuple[int, int]:
    road_pixels = int(np.sum(binary_mask == 255))
    total_pixels = int(binary_mask.size)
    return road_pixels, total_pixels


def process_split(
    split_name: str,
    raw_root: Path,
    output_root: Path,
    splits_root: Path,
    road_color: tuple[int, int, int],
) -> dict:
    """
    Convert one split: train, val, or test.
    Also creates a split text file containing image names.
    """
    image_dir = raw_root / split_name
    label_dir = raw_root / f"{split_name}_labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    output_mask_dir = output_root / "binary_masks" / split_name
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    splits_root.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    split_entries = []
    total_road_pixels = 0
    total_pixels = 0
    converted_count = 0
    missing_label_count = 0

    for image_path in image_files:
        # CamVid labels usually have the same filename but are stored in *_labels
        possible_label_paths = [
    label_dir / image_path.name,
    label_dir / image_path.with_suffix(".png").name,

    # Common CamVid Kaggle label naming convention
    label_dir / f"{image_path.stem}_L.png",
    label_dir / f"{image_path.stem}_L{image_path.suffix}",

    # Extra fallback names, in case another dataset version is used
    label_dir / f"{image_path.stem}_label.png",
    label_dir / f"{image_path.stem}_gt.png",
    label_dir / f"{image_path.stem}_mask.png",
]
        label_path = None
        for candidate in possible_label_paths:
            if candidate.exists():
                label_path = candidate
                break

        if label_path is None:
            print(f"[WARNING] Missing label for image: {image_path.name}")
            missing_label_count += 1
            continue

        binary_mask = convert_mask_to_binary(label_path, road_color)

        output_mask_path = output_mask_dir / image_path.with_suffix(".png").name
        Image.fromarray(binary_mask).save(output_mask_path)

        road_pixels, pixels = count_road_pixels(binary_mask)
        total_road_pixels += road_pixels
        total_pixels += pixels

        split_entries.append(image_path.name)
        converted_count += 1

    # Save split file
    split_file = splits_root / f"{split_name}.txt"
    with open(split_file, "w", encoding="utf-8") as f:
        for entry in split_entries:
            f.write(entry + "\n")

    road_ratio = total_road_pixels / total_pixels if total_pixels > 0 else 0.0

    return {
        "split": split_name,
        "images_found": len(image_files),
        "converted_masks": converted_count,
        "missing_labels": missing_label_count,
        "total_pixels": total_pixels,
        "road_pixels": total_road_pixels,
        "road_pixel_ratio": road_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CamVid binary road segmentation masks.")
    parser.add_argument("--raw-root", type=str, default="data/raw/CamVid", help="Path to raw CamVid folder")
    parser.add_argument("--output-root", type=str, default="data/processed", help="Path to processed data folder")
    parser.add_argument("--splits-root", type=str, default="data/splits", help="Path to save train/val/test split txt files")
    parser.add_argument("--road-class", type=str, default="Road", help="Class name used for the road class in class_dict.csv")

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    splits_root = Path(args.splits_root)

    class_dict_path = raw_root / "class_dict.csv"

    if not class_dict_path.exists():
        raise FileNotFoundError(f"class_dict.csv not found at: {class_dict_path}")

    print("Reading class dictionary...")
    road_color = find_road_color(class_dict_path, args.road_class)
    print(f"Road class color found: RGB{road_color}")

    summaries = []

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing split: {split_name}")
        summary = process_split(
            split_name=split_name,
            raw_root=raw_root,
            output_root=output_root,
            splits_root=splits_root,
            road_color=road_color,
        )
        summaries.append(summary)

        print(f"Images found: {summary['images_found']}")
        print(f"Converted masks: {summary['converted_masks']}")
        print(f"Missing labels: {summary['missing_labels']}")
        print(f"Road pixel ratio: {summary['road_pixel_ratio']:.4f}")

    output_root.mkdir(parents=True, exist_ok=True)

    dataset_summary = {
        "dataset": "CamVid",
        "task": "Binary road segmentation",
        "road_color_rgb": road_color,
        "mask_values": {
            "road": 255,
            "non_road": 0,
        },
        "splits": summaries,
    }

    summary_path = output_root / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=4)

    print("\nDataset preparation complete.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()