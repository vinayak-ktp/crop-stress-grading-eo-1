# Explainable Deep Learning for Crop Stress Grading

A comparison of four deep learning architectures for early detection of crop stress using hyperspectral satellite data, with a focus on **Explainable AI (XAI)** — mapping model predictions back to specific spectral bands to provide physically meaningful interpretations.

The central finding is that we can classify the stage of crop stress using EO-1 data while explaining _why_ the model made its decision, making the models significantly more trustworthy for agricultural practitioners or further domain-specific validation.

---

## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [Explainability — How It Works](#explainability--how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)

---

## Dataset

Hyperspectral observations extracted from EO-1 satellite data, containing metadata and detailed spectral reflectance values across various wavelength bands.

| Column   | Description                                  |
| -------- | -------------------------------------------- |
| `target` | Stage of crop stress — **prediction target** |
| `F_*`    | Processed spectral features & indices        |

**Preprocessing** (`scripts/prepare_data.py`):

- Removal of low-quality features (>50% missing values).
- Savitzky-Golay filtering for signal noise reduction.
- Derivative spectroscopy (1st derivative) extraction.
- Computation of specific vegetation indices (`MLVI`, `H_VSI`).
- Chronological/Stratified 70/15/15 split → `data/splits/{train,val,test}.csv`
- Normalisation using `StandardScaler` (Z-score) on all feature columns.
- Data augmentation applied via `SMOTE` on the training set to resolve class imbalance.

---

## Models

Four model architectures are implemented and compared, all trained to predict the categorical stress stage from the flat hyperspectral sequence.

| Model       | Architecture                                |
| ----------- | ------------------------------------------- |
| Paper CNN   | 1D Convolutional Neural Network baseline    |
| ResNet      | 1D ResNet with deep skip connections        |
| Transformer | Vision Transformer adapted for 1D sequences |
| Hybrid      | CNN and Transformer blocks combined         |

### Paper CNN (`models/paper_cnn.py`)

A baseline 1D Convolutional Neural Network that uses local receptive fields to aggregate spectral information from adjacent wavelengths.

### ResNet (`models/resnet.py`)

A 1D implementation of Residual Networks. It incorporates skip connections to allow for deeper architectures, enabling the extraction of highly complex non-linear combinations of spectral bands without suffering from vanishing gradients.

### Transformer (`models/transformer.py`)

A model relying entirely on self-attention mechanisms. It projects the 1D hyperspectral input into a higher dimension and applies multi-head attention blocks, allowing every spectral band to dynamically attend to every other band, regardless of distance in the sequence.

### Hybrid (`models/hybrid.py`)

Combines the local feature extraction strengths of 1D Convolutions with the global contextual understanding of Transformers.

---

## Explainability — How It Works

Standard neural networks operate as "black boxes", outputting a stress grade without justifying which parts of the electromagnetic spectrum contributed to the decision. This project implements multiple XAI techniques to explicitly map decision importance.

### Techniques

1. **Grad-CAM**: Computes the gradient of the target class with respect to the final convolutional layer to produce a coarse localization map, highlighting the most important spectral regions.
2. **GradientShap**: Approximates SHAP values by computing the expectations of gradients for the inputs across a manifold of background data, offering robust and pixel-level attribution.
3. **Attention Weights**: For Transformer-based models, extracts raw self-attention patterns indicating which bands the model naturally focused on most.

### Why This Helps

Hyperspectral data contains hundreds of bands, many of which may be redundant or irrelevant. By deriving "Top Spectral Bands by Mean Attribution", domain experts can verify if the models are focusing on physically meaningful absorption features (e.g., chlorophyll absorption at ~680 nm, or moisture content in the SWIR range), verifying that the model relies on true signal, rather than dataset artifacts.

---

## Results

All models are trained with the AdamW optimiser, cosine annealing learning rate scheduler with warm restarts, cross-entropy loss with label smoothing, and early stopping.

| Model       | Accuracy (%) | F1 Score (%) |
| ----------- | ------------ | ------------ |
| Paper CNN   | 83.85        | 83.88        |
| ResNet      | 85.96        | 86.00        |
| Transformer | 88.17        | 88.19        |
| Hybrid      | 90.48        | 90.48        |

---

## Project Structure

```
crop-stress-grading/
│
├── data/
│   └── splits/                    # output of scripts/prepare_data.py
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── data_pipeline/
│   ├── __init__.py
│   ├── dataloader.py              # dataset & loaders
│   └── preprocessing.py           # filtering, SMOTE, MLVI derivation
│
├── models/
│   ├── __init__.py
│   ├── paper_cnn.py
│   ├── resnet.py
│   ├── transformer.py
│   └── hybrid.py
│
├── explainability/
│   ├── __init__.py
│   ├── attention_viz.py
│   ├── grad_cam.py
│   └── gradient_shap.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # train_one_epoch, evaluate, train
│   └── metrics.py                 # get_predictions, compute_metrics
│
├── scripts/
│   ├── prepare_data.py            # run once to generate data/splits/
│   ├── train.py                   # main training entry point
│   ├── explain.py                 # XAI visualizer map generator
│   ├── plot_results.py            # generate plots from saved results
│   └── analyze_data.py            # exploratory data analysis on raw data
│
├── checkpoints/                   # saved model weights, organised by experiment
│
├── _results/
│   ├── raw/                       # per-model JSON results
│   ├── plots/                     # generated plots
│   └── explain/                   # saved salience and heat maps
│
├── install.py                     # intelligent setup script
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/vinayak-ktp/crop-stress-grading-eo-1.git
cd crop-stress-grading-eo-1

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# install dependencies automatically (detects GPU for PyTorch)
python install.py
```

---

## Usage

### 1. Prepare data

Cleans, applies Savitzky-Golay filtering, generates MLVI/HVSI indices, normalizes, applies SMOTE, and saves training splits:

```bash
python -m scripts.prepare_data
```

Output: `data/splits/train.csv`, `val.csv`, `test.csv`

### 2. Train a single model

train.py allows you to train a single model or all models at once.

```bash
python -m scripts.train --model hybrid --exp MyExperiment
```

**--model** expects: `paper_cnn`, `resnet`, `transformer`, `hybrid` or `all`

**--exp** sets the subdirectory name under `checkpoints/` and `_results/raw/`

### 3. Generate Explainability Maps

Generate salience and interaction graphs for model interpretation.

```bash
python -m scripts.explain --exp MyExperiment --model hybrid --method all
```

Output: Multiple heatmaps and graphs in `_results/explain/MyExperiment/hybrid/`

### 4. Generate Performance Plots

Reads all metrics JSONs from `_results/raw/MyExperiment` and renders comparative plots.

```bash
python -m scripts.plot_results --exp MyExperiment
```
