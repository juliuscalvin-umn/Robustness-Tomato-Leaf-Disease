# Robustness Analysis of EfficientNet-B0 and FastViT-T8 for Tomato Leaf Disease Classification under Individual Visual Augmentation

Official implementation of the undergraduate thesis:

**Comparative Robustness Analysis of EfficientNet-B0 and FastViT-T8 for Tomato Leaf Disease Classification under Individual Visual Augmentation**

📄 **Paper:** Under Review

---

# 📋 Overview

This repository contains the implementation of a comparative study between **EfficientNet-B0** and **FastViT-T8** for tomato leaf disease classification under various visual perturbations.

Unlike conventional image classification studies that only evaluate clean-image accuracy, this research investigates model robustness by applying individual visual augmentations with different severity levels. The evaluation measures each model's ability to maintain classification performance under realistic environmental conditions.

The project is implemented using **PyTorch** and includes model training, evaluation, robustness testing, visualization, and performance comparison.

## Key Features

- Comparison between EfficientNet-B0 and FastViT-T8
- Individual visual augmentation evaluation
- Multiple severity levels for each augmentation
- Robustness analysis using various evaluation metrics
- Automatic confusion matrix generation
- Classification report generation
- Confidence and inference latency analysis
- AURC (Area Under Robustness Curve) evaluation
- PyTorch implementation

---

# 🎯 Key Contributions

- Comprehensive robustness comparison between EfficientNet-B0 and FastViT-T8.
- Individual augmentation evaluation to analyze the effect of each visual variation independently.
- Robustness evaluation using multiple quantitative metrics including:
  - Mean Accuracy
  - Worst Accuracy
  - Mean Drop
  - Worst Drop
  - Robustness Score
  - Standard Deviation of Accuracy Drop
  - Average Confidence
  - Average Latency
  - Area Under Robustness Curve (AURC)
- Analysis of model performance under realistic visual degradation commonly encountered in agricultural environments.

---

# Project Architecture / Research Workflow

<p align="center">
<img src="images/workflow.png" width="900">
</p>

---

# 🚀 Installation

## Requirements

- Python 3.11
- PyTorch
- torchvision
- timm
- numpy
- pandas
- matplotlib
- scikit-learn
- OpenCV
- Pillow
- tqdm

## Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset Preparation

## Tomato Leaf Disease Dataset

The dataset used in this research is publicly available on Kaggle.

### Download

**Dataset Link:**

https://www.kaggle.com/datasets/juliusiota/tomato-leaf-disease

or visit:

https://www.kaggle.com/datasets/juliusiota/tomato-leaf-disease

The dataset consists of tomato leaf images belonging to ten disease categories and one healthy class. Images were collected, standardized, cleaned, and organized into training, validation, and testing subsets for robustness evaluation. :contentReference[oaicite:0]{index=0}

### Dataset Structure

```
dataset/

├── train/
│   ├── Bacterial_Spot/
│   ├── Early_Blight/
│   ├── Healthy/
│   ├── Late_Blight/
│   ├── Leaf_Mold/
│   ├── Septoria_Leaf_Spot/
│   ├── Spider_Mites/
│   ├── Target_Spot/
│   ├── Tomato_Mosaic_Virus/
│   └── Tomato_Yellow_Leaf_Curl_Virus/
│
├── validation/
│   └── ...
│
└── test/
    └── ...
```

---

# 🏋️ Training

Run the notebook sequentially.

Or execute the training script (if available):

```bash
python train.py
```

Example hyperparameters:

| Parameter | Value |
|------------|------:|
| Image Size | 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 50 |
| Loss Function | CrossEntropyLoss |

---


# 📊 Results

The proposed robustness evaluation compares **EfficientNet-B0** and **FastViT-T8** under seven individual visual augmentations with multiple severity levels. Both models achieved the same clean-image classification accuracy of **99.88%**. The robustness evaluation was then conducted using the Robustness Score, Mean Accuracy Drop, and Mean Latency.

## Clean Dataset Performance

| Model | Clean Accuracy |
|--------|---------------:|
| EfficientNet-B0 | **99.88%** |
| FastViT-T8 | **99.88%** |

---

## Overall Robustness Evaluation

| Model | Augmentation | Robustness Score ↓ | Mean Accuracy Drop ↓ | Mean Latency ↑ |
|--------|--------------|-------------------:|---------------------:|---------------:|
| EfficientNet-B0 | Brightness | 15.071% | 60.706% | 84.815% |
| FastViT-T8 | Brightness | **12.215%** | **47.756%** | **87.685%** |
| EfficientNet-B0 | Contrast | **6.041%** | **14.155%** | **93.844%** |
| FastViT-T8 | Contrast | 6.163% | 16.435% | 93.737% |
| EfficientNet-B0 | Saturation | 1.398% | 3.614% | 98.487% |
| FastViT-T8 | Saturation | **0.964%** | **2.495%** | **98.935%** |
| EfficientNet-B0 | Hue | 23.727% | 54.180% | 76.158% |
| FastViT-T8 | Hue | **19.787%** | **46.451%** | **80.113%** |
| EfficientNet-B0 | Affine | **0.197%** | **0.645%** | **99.688%** |
| FastViT-T8 | Affine | 0.301% | 0.774% | 99.599% |
| EfficientNet-B0 | Perspective | **20.823%** | **56.188%** | **79.062%** |
| FastViT-T8 | Perspective | 42.059% | 88.943% | 57.841% |
| EfficientNet-B0 | Blur | 12.229% | 18.643% | 87.656% |
| FastViT-T8 | **8.666%** | **13.266%** | **91.234%** |

> **Lower Robustness Score and Mean Accuracy Drop indicate better robustness, while higher Mean Latency (%) indicates better efficiency retention relative to the clean condition.**

---

## Summary

- Both **EfficientNet-B0** and **FastViT-T8** achieved **99.88%** classification accuracy on the clean test dataset.
- **FastViT-T8** demonstrated superior robustness under **Brightness, Saturation, Hue, and Blur** perturbations.
- **EfficientNet-B0** showed better robustness under **Contrast, Affine, and Perspective** transformations.
- Among all evaluated perturbations, **Affine transformation** had the smallest impact on both models, while **Perspective distortion** produced the largest performance degradation, particularly for FastViT-T8.

# 🏗️ Project Structure

```
Project/

├── Robustness_Analysis.ipynb
├── README.md
├── requirements.txt
├── images/
│   ├── workflow.png
│   ├── confusion_matrix.png
│   └── robustness_curve.png
└── results/
```

---



# 🙏 Acknowledgments

This research was conducted under the supervision of:

- Big Data Laboratory
- Information Systems Study Program
- Universitas Multimedia Nusantara

Special thanks to all publicly available tomato leaf disease dataset providers.

---



# 📜 License

This project is released for academic and research purposes.
