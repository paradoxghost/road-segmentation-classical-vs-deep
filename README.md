# Road Segmentation: Classical Vision vs Deep Learning

This project compares classical computer vision methods and a lightweight deep learning model for binary road segmentation on the CamVid dataset.

The original CamVid semantic segmentation masks are converted into a binary task:

- Road pixels: `255`
- Non-road pixels: `0`

## Project Goal

The goal is to study road-region segmentation as a comparative computer vision problem. The project will evaluate classical segmentation baselines against a small U-Net model using segmentation metrics such as IoU, Dice, precision, recall, and pixel accuracy.

## Planned Methods

Classical vision baselines:

- HSV/color thresholding
- Otsu thresholding
- K-means clustering
- Morphological post-processing

Deep learning baseline:

- Lightweight U-Net
- Binary mask output
- Evaluation with IoU, Dice, precision, recall, and pixel accuracy

## Dataset Preparation

The project uses the CamVid semantic segmentation dataset. The dataset is not included in this repository because it is large; place it locally in:

```text
data/raw/CamVid/
|-- train/
|-- train_labels/
|-- val/
|-- val_labels/
|-- test/
|-- test_labels/
`-- class_dict.csv
```

The `data/` directory is ignored by Git and should not be committed to GitHub.

Prepared split:

| Split | Images | Road Pixel Ratio |
|---|---:|---:|
| Train | 369 | 28.54% |
| Validation | 100 | 28.78% |
| Test | 232 | 24.65% |

To prepare the binary road/non-road masks:

```bash
python src/prepare_data.py
```

To verify the prepared dataset and generate dataset exploration figures:

```bash
python src/explore_dataset.py
```

Generated figures are saved in:

```text
results/figures/dataset_examples/
```

## Classical Baselines

Step 5 implements four classical segmentation baselines:

- HSV/color thresholding
- Otsu thresholding
- K-means clustering without morphology
- K-means clustering with morphology

Run all classical baselines on a split:

```bash
python src/classical/run_classical_baselines.py --split val
```

Prediction masks are saved in:

```text
results/predictions/hsv/{split}/
results/predictions/otsu/{split}/
results/predictions/kmeans_raw/{split}/
results/predictions/kmeans_morph/{split}/
```

Metrics JSON files are saved in:

```text
results/metrics/classical/
```

Visual comparison figures are saved in:

```text
results/figures/classical/{split}/
```

Validation results from the classical baselines:

| Method | Pixel Acc | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| HSV | 0.6773 | 0.4674 | 0.6370 | 0.4710 | 0.9839 |
| Otsu | 0.7348 | 0.4969 | 0.6639 | 0.5225 | 0.9105 |
| K-means Raw | 0.8744 | 0.6413 | 0.7814 | 0.7825 | 0.7804 |
| K-means Morph | 0.8750 | 0.6432 | 0.7829 | 0.7824 | 0.7834 |

## Deep Learning Baseline

Step 6 implements a lightweight U-Net for binary road segmentation.

Train on the training split, validate on the validation split, generate validation predictions, and evaluate them:

```bash
python src/deep/run_unet_baseline.py --epochs 20 --batch-size 8 --image-size 256
```

The best model checkpoint is saved in:

```text
results/models/unet_best.pt
```

Training history is saved in:

```text
results/metrics/deep/unet_history.json
```

Validation predictions and metrics are saved in:

```text
results/predictions/unet/val/
results/metrics/deep/unet_val.json
```

Validation comparison figures are saved in:

```text
results/figures/deep/val/
```

U-Net validation result:

| Method | Pixel Acc | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9743 | 0.9137 | 0.9549 | 0.9638 | 0.9462 |

## Model Weights

The trained U-Net checkpoint is stored at:

```text
results/models/unet_best.pt
```

Because this is a large binary file, it is tracked with Git LFS. If the checkpoint is not available after cloning, install Git LFS and pull LFS files:

```bash
git lfs install
git lfs pull
```

The checkpoint can also be reproduced by training:

```bash
python src/deep/run_unet_baseline.py --epochs 20 --batch-size 8 --image-size 256
```

## Final Evaluation

After method design is fixed, evaluate on the test split without further tuning.

Run classical baselines on the test split:

```bash
python src/classical/run_classical_baselines.py --split test
```

Evaluate the saved U-Net checkpoint on the test split:

```bash
python src/deep/evaluate_unet.py --split test --image-size 256 --checkpoint results/models/unet_best.pt
```

Predicted masks are saved in:

```text
results/predictions/unet/test/
```

Create final validation/test comparison tables and charts:

```bash
python src/compare_results.py
```

Create best/worst and failure-case figures:

```bash
python src/failure_cases.py
```

Final comparison outputs are saved in:

```text
results/metrics/final_comparison_val.csv
results/metrics/final_comparison_test.csv
results/metrics/final_comparison_summary.json
results/figures/final_comparison/
results/figures/failure_cases/
```

Final test results:

| Method | Pixel Acc | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| HSV | 0.6792 | 0.4205 | 0.5920 | 0.4312 | 0.9440 |
| Otsu | 0.7333 | 0.4521 | 0.6227 | 0.4781 | 0.8926 |
| K-means Raw | 0.8679 | 0.6050 | 0.7539 | 0.6970 | 0.8209 |
| K-means Morph | 0.8678 | 0.6058 | 0.7545 | 0.6959 | 0.8239 |
| U-Net | 0.9723 | 0.8912 | 0.9425 | 0.9639 | 0.9220 |

## Demo App

To launch the local software demo:

```bash
python -m streamlit run app/demo_app.py
```

The demo includes 30 bundled CamVid test images in:

```text
app/sample_images/
```

These samples let the app run immediately without downloading the full CamVid dataset. They are selected from across the test split so the model can be tried on multiple road scenes. The full dataset remains excluded from Git and Docker because it is large.

## One-command Docker Demo

The easiest way to test the demo is to use the prebuilt Docker image. This does not require cloning the repository, installing Python dependencies, or downloading the CamVid dataset. The Docker image already includes the Streamlit demo, the trained U-Net checkpoint, and bundled sample road images.

Run:

```bash
docker run --rm -p 8501:8501 vitosparadox/road-segmentation-demo:latest
```

Then open:

```text
http://localhost:8501
```

If port 8501 is busy:

```bash
docker run --rm -p 8502:8501 vitosparadox/road-segmentation-demo:latest
```

Then open:

```text
http://localhost:8502
```

Note: Docker will automatically download the prebuilt image the first time. No dataset download is required for the demo.

Docker option:

```bash
docker build -t road-seg-demo .
docker run --rm -p 8501:8501 road-seg-demo
```

Then open:

```text
http://localhost:8501
```

If port `8501` is already in use:

```bash
docker run --rm -p 8502:8501 road-seg-demo
```

Then open:

```text
http://localhost:8502
```

## Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
road-segmentation-classical-vs-deep/
|-- README.md
|-- requirements.txt
|-- data/
|-- notebooks/
|-- results/
|-- src/
`-- report/
```

## Current Status

- [x] Project structure created
- [x] CamVid masks converted to binary road/non-road masks
- [x] Dataset exploration and visualization added
- [x] Classical baselines evaluated
- [x] U-Net trained and evaluated on validation split
- [x] Final test evaluation completed
- [x] Final report and demo video completed
