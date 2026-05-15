"""Visualization helpers for image, mask, prediction, and error maps."""

from pathlib import Path
import matplotlib.pyplot as plt


def save_comparison(image, gt_mask, pred_mask, save_path: str | Path, title: str = "Comparison") -> None:
    """Save a simple visual comparison figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow((gt_mask > 0) != (pred_mask > 0), cmap="gray")
    axes[3].set_title("Error map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
