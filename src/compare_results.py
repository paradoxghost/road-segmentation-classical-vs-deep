"""Create final validation/test comparison tables and bar charts."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "final_comparison"

METHOD_FILES = [
    ("HSV", "classical/hsv_{split}.json"),
    ("Otsu", "classical/otsu_{split}.json"),
    ("KMeans Raw", "classical/kmeans_raw_{split}.json"),
    ("KMeans Morph", "classical/kmeans_morph_{split}.json"),
    ("U-Net", "deep/unet_{split}.json"),
]

METRIC_COLUMNS = ["pixel_accuracy", "iou", "dice", "precision", "recall"]
DISPLAY_COLUMNS = {
    "pixel_accuracy": "Pixel Acc",
    "iou": "IoU",
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Recall",
}


def load_json(path: Path) -> dict:
    """Load one metrics JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {path}. "
            "Run the validation/test evaluation commands before comparing results."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_split_metrics(split: str) -> pd.DataFrame:
    """Collect all method metrics for one split into a DataFrame."""
    rows = []

    for method_name, template in METHOD_FILES:
        metrics_path = METRICS_ROOT / template.format(split=split)
        metrics = load_json(metrics_path)

        row = {"Method": method_name}
        for metric_name in METRIC_COLUMNS:
            row[DISPLAY_COLUMNS[metric_name]] = float(metrics[metric_name])
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(title: str, dataframe: pd.DataFrame) -> None:
    """Print a readable metrics table."""
    print()
    print(title)
    print(f"{'Method':<15} {'Pixel Acc':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 64)

    for _, row in dataframe.iterrows():
        print(
            f"{row['Method']:<15} "
            f"{row['Pixel Acc']:>10.4f} "
            f"{row['IoU']:>8.4f} "
            f"{row['Dice']:>8.4f} "
            f"{row['Precision']:>10.4f} "
            f"{row['Recall']:>8.4f}"
        )


def save_metric_chart(dataframe: pd.DataFrame, split: str, metric: str, output_path: Path) -> None:
    """Save a simple bar chart for one metric."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(dataframe["Method"], dataframe[metric], color=["#6b7280", "#9ca3af", "#60a5fa", "#2563eb", "#16a34a"])

    ax.set_title(f"{split.capitalize()} {metric} Comparison")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 0.02, 0.98),
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dataframe_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert a display DataFrame to JSON-friendly records."""
    return [
        {
            "method": row["Method"],
            "pixel_accuracy": float(row["Pixel Acc"]),
            "iou": float(row["IoU"]),
            "dice": float(row["Dice"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
        }
        for _, row in dataframe.iterrows()
    ]


def main() -> None:
    validation_df = collect_split_metrics("val")
    test_df = collect_split_metrics("test")

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    validation_csv = METRICS_ROOT / "final_comparison_val.csv"
    test_csv = METRICS_ROOT / "final_comparison_test.csv"
    summary_json = METRICS_ROOT / "final_comparison_summary.json"

    validation_df.to_csv(validation_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    summary = {
        "validation": dataframe_records(validation_df),
        "test": dataframe_records(test_df),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    save_metric_chart(validation_df, "validation", "IoU", FIGURES_ROOT / "validation_iou_comparison.png")
    save_metric_chart(validation_df, "validation", "Dice", FIGURES_ROOT / "validation_dice_comparison.png")
    save_metric_chart(test_df, "test", "IoU", FIGURES_ROOT / "test_iou_comparison.png")
    save_metric_chart(test_df, "test", "Dice", FIGURES_ROOT / "test_dice_comparison.png")

    print_table("VALIDATION RESULTS", validation_df)
    print_table("TEST RESULTS", test_df)

    print()
    print(f"Saved validation CSV to {validation_csv.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved test CSV to {test_csv.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved summary JSON to {summary_json.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved final comparison figures to {FIGURES_ROOT.relative_to(PROJECT_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
