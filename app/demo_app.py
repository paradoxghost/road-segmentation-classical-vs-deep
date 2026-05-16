"""Streamlit demo for road-region segmentation on CamVid."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep src imports stable when Streamlit is launched from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep.predict import load_model


MODEL_PATH = PROJECT_ROOT / "results" / "models" / "unet_best.pt"
FINAL_COMPARISON_DIR = PROJECT_ROOT / "results" / "figures" / "final_comparison"
CLASSICAL_TEST_DIR = PROJECT_ROOT / "results" / "figures" / "classical" / "test"
DEEP_TEST_DIR = PROJECT_ROOT / "results" / "figures" / "deep" / "test"
FAILURE_CASES_DIR = PROJECT_ROOT / "results" / "figures" / "failure_cases"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "app" / "sample_images"
IMAGE_SIZE = 256
THRESHOLD = 0.5


st.set_page_config(
    page_title="Road Region Segmentation Demo",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
            color: #172033;
        }
        .stApp p, .stApp label, .stApp span {
            color: #172033;
        }
        [data-testid="stSidebar"] {
            background: #eaf1f8;
            border-right: 1px solid #d7e1ec;
        }
        [data-testid="stSidebar"] * {
            color: #172033 !important;
        }
        h1, h2, h3 {
            color: #123b63;
        }
        .intro-box {
            background: #ffffff;
            border: 1px solid #d9e4ef;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 1.25rem;
        }
        .intro-box, .intro-box * {
            color: #172033 !important;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #d9e4ef;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }
        .metric-card, .metric-card * {
            color: #172033 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px dashed #8ab2d6 !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #172033 !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: #123b63 !important;
            border: 1px solid #123b63 !important;
            color: #ffffff !important;
        }
        [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }
        [data-testid="stAlert"] {
            background: #d8ebff !important;
            border: 1px solid #a9cfee !important;
        }
        [data-testid="stAlert"] * {
            color: #123b63 !important;
        }
        .small-muted {
            color: #5e6b7a;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    """Render the shared project title and short explanation."""
    st.title("Road Region Segmentation Demo")
    st.subheader("Classical Vision vs U-Net on CamVid")
    st.markdown(
        """
        <div class="intro-box">
            This demo predicts road vs non-road pixels from a road-scene image.<br>
            <strong>White = road.</strong><br>
            <strong>Black = non-road.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render navigation and fixed project metadata."""
    st.sidebar.title("Demo Controls")
    mode = st.sidebar.radio(
        "Choose mode",
        ["U-Net inference", "Project results", "Failure cases"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Project Info")
    st.sidebar.markdown(
        """
        <div class="metric-card"><strong>Dataset:</strong> CamVid</div>
        <div class="metric-card"><strong>Train:</strong> 369</div>
        <div class="metric-card"><strong>Validation:</strong> 100</div>
        <div class="metric-card"><strong>Test:</strong> 232</div>
        <div class="metric-card"><strong>Model:</strong> Lightweight U-Net trained from scratch</div>
        <div class="metric-card"><strong>Test IoU:</strong> 0.8912</div>
        <div class="metric-card"><strong>Test Dice:</strong> 0.9425</div>
        """,
        unsafe_allow_html=True,
    )
    return mode


@st.cache_resource(show_spinner="Loading U-Net checkpoint...")
def get_unet_model() -> tuple[torch.nn.Module, torch.device]:
    """Load the trained U-Net once per Streamlit session."""
    # Cache the checkpoint so widget changes do not reload the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(model_path=MODEL_PATH, device=device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize and convert a PIL image to the tensor format used during training."""
    resized_image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.BILINEAR,
    )
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return image_tensor


def predict_mask(image: Image.Image) -> np.ndarray:
    """Run U-Net inference and return a 0/255 binary mask."""
    model, device = get_unet_model()
    image_tensor = preprocess_image(image).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= THRESHOLD).squeeze().cpu().numpy().astype(np.uint8) * 255

    return mask


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Blend the road mask over the uploaded image for visual inspection."""
    base_image = image.convert("RGB")
    display_mask = Image.fromarray(mask).resize(base_image.size, resample=Image.NEAREST)
    mask_array = np.asarray(display_mask) > 127

    # Highlight predicted road pixels in blue while preserving image context.
    base_array = np.asarray(base_image, dtype=np.float32)
    road_color = np.array([0, 112, 192], dtype=np.float32)
    base_array[mask_array] = (0.6 * base_array[mask_array]) + (0.4 * road_color)

    return Image.fromarray(np.clip(base_array, 0, 255).astype(np.uint8))


def image_paths(directory: Path, limit: int | None = None) -> list[Path]:
    """Return sorted displayable image paths from a results directory."""
    if not directory.exists():
        return []

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if limit is None:
        return paths
    return paths[:limit]


def render_unet_inference() -> None:
    """Render the single-image U-Net inference demo."""
    st.header("U-Net Inference")
    st.write(
        "The U-Net predicts a probability map for road pixels, then the output is "
        "thresholded into a binary mask."
    )

    sample_images = image_paths(SAMPLE_IMAGE_DIR)
    selected_sample: Path | None = None
    if sample_images:
        selected_sample_name = st.selectbox(
            "Use a bundled CamVid test sample",
            [path.name for path in sample_images],
        )
        selected_sample = SAMPLE_IMAGE_DIR / selected_sample_name

    uploaded_file = st.file_uploader(
        "Or upload a road-scene image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_caption = uploaded_file.name
    elif selected_sample is not None:
        input_image = Image.open(selected_sample).convert("RGB")
        image_caption = selected_sample.name
    else:
        st.info("Choose a bundled CamVid sample or upload a JPG/PNG image to run the trained U-Net model.")
        return

    try:
        mask = predict_mask(input_image)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except RuntimeError as exc:
        st.error(f"Could not run U-Net inference: {exc}")
        return

    overlay = make_overlay(input_image, mask)
    mask_image = Image.fromarray(mask)

    col_input, col_mask, col_overlay = st.columns(3)
    with col_input:
        st.subheader("Input image")
        st.image(input_image, caption=image_caption, use_container_width=True)
    with col_mask:
        st.subheader("Predicted road mask")
        st.image(mask_image, clamp=True, use_container_width=True)
    with col_overlay:
        st.subheader("Overlay on original image")
        st.image(overlay, use_container_width=True)


def render_project_results() -> None:
    """Render final quantitative and visual project results."""
    st.header("Project Results")

    results = pd.DataFrame(
        [
            ["HSV", 0.6792, 0.4205, 0.5920, 0.4312, 0.9440],
            ["Otsu", 0.7333, 0.4521, 0.6227, 0.4781, 0.8926],
            ["K-means Raw", 0.8679, 0.6050, 0.7539, 0.6970, 0.8209],
            ["K-means Morph", 0.8678, 0.6058, 0.7545, 0.6959, 0.8239],
            ["U-Net", 0.9723, 0.8912, 0.9425, 0.9639, 0.9220],
        ],
        columns=["Method", "Pixel Acc", "IoU", "Dice", "Precision", "Recall"],
    )
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.subheader("Final Comparison Charts")
    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.image(
            FINAL_COMPARISON_DIR / "test_iou_comparison.png",
            caption="Test IoU comparison",
            use_container_width=True,
        )
    with chart_col_2:
        st.image(
            FINAL_COMPARISON_DIR / "test_dice_comparison.png",
            caption="Test Dice comparison",
            use_container_width=True,
        )

    st.subheader("Example Test Figures")
    classical_examples = image_paths(CLASSICAL_TEST_DIR, limit=1)
    unet_examples = image_paths(DEEP_TEST_DIR, limit=1)

    example_col_1, example_col_2 = st.columns(2)
    with example_col_1:
        if classical_examples:
            st.image(
                classical_examples[0],
                caption="Classical methods comparison",
                use_container_width=True,
            )
        else:
            st.warning(f"No classical test figures found in {CLASSICAL_TEST_DIR}")
    with example_col_2:
        if unet_examples:
            st.image(
                unet_examples[0],
                caption="U-Net prediction example",
                use_container_width=True,
            )
        else:
            st.warning(f"No U-Net test figures found in {DEEP_TEST_DIR}")


def render_failure_cases() -> None:
    """Render saved failure-case examples."""
    st.header("Failure Cases")
    st.write(
        "Failure cases mainly occur with shadows, low contrast, sidewalks similar to "
        "roads, occlusions, and ambiguous road boundaries."
    )

    paths = image_paths(FAILURE_CASES_DIR, limit=9)
    if not paths:
        st.warning(f"No failure-case figures found in {FAILURE_CASES_DIR}")
        return

    columns = st.columns(3)
    for index, path in enumerate(paths):
        with columns[index % 3]:
            st.image(path, caption=path.stem.replace("_", " ").title(), use_container_width=True)


def main() -> None:
    render_header()
    mode = render_sidebar()

    if mode == "U-Net inference":
        render_unet_inference()
    elif mode == "Project results":
        render_project_results()
    else:
        render_failure_cases()


if __name__ == "__main__":
    main()
