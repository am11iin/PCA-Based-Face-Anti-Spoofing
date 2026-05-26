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

> ⚠️ **Note:** Please update this section to confirm the exact dataset used.

This project is designed to work with a face anti-spoofing dataset structured into two classes:

| Class | Description |
|-------|-------------|
| `real` / `client` | Authentic live face images |
| `fake` / `imposter` | Spoofed face images (printed photos or replayed videos) |

A commonly used dataset for this type of pipeline is the **NUAA Imposter Database**:

- **Name:** NUAA Face Anti-Spoofing Dataset
- **Source:** [NUAA — Nanjing University of Aeronautics and Astronautics](http://parnec.nuaa.edu.cn/xtan/NUAAImposterDB_download.html)
- **Also available on Kaggle:** [nuaaaa dataset](https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa)
- **Content:** ~5,000 real face images + ~5,000 imposter (printed photo) images
- **Format:** JPEG/PNG, grayscale or RGB

> If a different dataset was used (e.g., a custom dataset or another benchmark), please update this section accordingly.

### Expected folder structure

```
dataset/
├── ClientFace/       # Real face images
│   ├── 0001_001.jpg
│   └── ...
└── ImposterFace/     # Fake / spoofed face images
    ├── 0001_001.jpg
    └── ...
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
DATASET_PATH = "./dataset/"
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
