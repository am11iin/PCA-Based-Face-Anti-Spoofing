# 🎭 PCA-Based Face Anti-Spoofing

> A machine learning pipeline for **real vs. fake face classification** using **PCA (Eigenfaces)** and **SVM**, with image preprocessing, threshold calibration, and full performance evaluation.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)

---

## Overview

This project implements a classic machine learning approach to **face anti-spoofing** — the task of distinguishing a **real (live) face** from a **fake (spoofed) one** (photo printout, screen replay, etc.).

The method is based on **Principal Component Analysis (PCA)** for dimensionality reduction and feature extraction (Eigenfaces), combined with a **Support Vector Machine (SVM)** classifier. The full pipeline covers:

- Image loading and preprocessing
- Feature normalization and PCA projection
- Eigenface visualization and analysis
- SVM training and threshold calibration
- Performance evaluation (accuracy, confusion matrix, ROC)

---

## Dataset

This project uses the **CASIA-FASD** (CASIA Face Anti-Spoofing Database), available on Kaggle.

| Field | Details |
|-------|---------|
| **Name** | CASIA Face Anti-Spoofing Database (CASIA-FASD) |
| **Source (Kaggle)** | [kaggle.com/datasets/minhnh2107/casiafasd](https://www.kaggle.com/datasets/minhnh2107/casiafasd) |
| **Original paper** | Z. Zhang et al., *"A Face Anti-Spoofing Database with Diverse Attacks"*, ICB 2012 |
| **Attack types** | Warped printed photos, printed photos with cut eyes, video replay |
| **Camera qualities** | Low quality, normal quality, high quality |
| **Classes** | `real` (live face) / `fake` (spoof attack) |
| **Format** | AVI videos → extracted frames (JPEG/PNG) |

### Dataset folder structure

```
CASIA_FASD/
├── train_release/
│   ├── 1/              # Real face videos/frames
│   └── 2/              # Fake face videos/frames (attack types)
└── test_release/
    ├── 1/              # Real
    └── 2/              # Fake
```

---

## Pipeline

```
Raw Images
    │
    ▼
┌──────────────────┐
│ Preprocessing    │  Resize, grayscale, flatten
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Normalization    │  Zero-mean, unit-variance per pixel
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ PCA              │  Eigenface extraction, dimensionality reduction
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ SVM Classifier   │  Binary classification: real vs. fake
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Threshold        │  Decision score calibration
│ Calibration      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Evaluation       │  Accuracy, Confusion Matrix, ROC/AUC
└──────────────────┘
```

---

## Features

- **Image Preprocessing** — resizing, grayscale conversion, pixel normalization
- **PCA / Eigenfaces** — extracts the most discriminative face components
- **Normalized PCA** — applies standardization before projection for improved stability
- **SVM Classification** — linear or RBF kernel SVM for binary classification
- **Threshold Calibration** — fine-tunes the decision boundary to balance FAR/FRR
- **Performance Evaluation** — accuracy, precision, recall, F1, confusion matrix, and ROC curve

---

## Project Structure

```
PCA-Based-Face-Anti-Spoofing/
├── Face_Anti_Spoofing_with_PCA.ipynb   # Main notebook — full pipeline
├── rapport projet AnaD.pdf             # Project report (Data Analysis course)
└── README.md
```

---

## Installation

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install numpy matplotlib scikit-learn opencv-python pillow
```

Or using a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Recommended: use a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install numpy matplotlib scikit-learn opencv-python pillow
```

---

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/am11iin/PCA-Based-Face-Anti-Spoofing.git
cd PCA-Based-Face-Anti-Spoofing
```

2. **Download and place the dataset** in a `dataset/` folder following the structure above.

3. **Open the notebook:**

```bash
jupyter notebook Face_Anti_Spoofing_with_PCA.ipynb
```

4. **Update the dataset path** in the first cell of the notebook if needed:

```python
DATASET_PATH = "./CASIA_FASD/"
```

5. **Run all cells** sequentially (Kernel → Restart & Run All).

---

## Results

> ⚠️ Update this section with your actual results after running the notebook.

| Metric | Value |
|--------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| AUC-ROC | — |

Key outputs include:
- Eigenface visualizations (top principal components)
- Confusion matrix (real vs. fake predictions)
- ROC curve with optimal threshold
- Classification report per class

---

## Authors

**Mohamed Amine Sellami**
M1 Computer Science — Data Analysis Project, 2025/2026

---

## License

This project was developed as an academic assignment. All rights reserved by the author.
