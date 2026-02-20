# Glaucoma Detection with DenseNet121

This repository contains a deep learning–based glaucoma detection model trained on retinal fundus images using **DenseNet121**.

## 📌 Project Overview
- **Task:** Binary classification (Glaucoma vs Normal)
- **Model:** DenseNet121 (ImageNet pretrained)
- **Framework:** PyTorch
- **Training Environment:** Google Colab

The model was trained and evaluated using a **stratified train/validation/test split** to prevent class imbalance and data leakage.

## 📊 Dataset
- Total images: **705**
  - Glaucoma: 396
  - Normal: 309
- Split ratio:
  - Train: 80%
  - Validation: 10%
  - Test: 10%

> Dataset images are **not included** in this repository.

## 🧪 Evaluation Results (Test Set)
- **Accuracy:** 1.00
- **Precision:** 1.00
- **Recall (Sensitivity):** 1.00
- **F1-score:** 1.00
- **ROC-AUC:** 1.00

Confusion Matrix:
[[31  0]
 [ 0 40]]

No data leakage or duplicate images were detected between splits.

## 📁 Repository Structure
.
├── densenet121_best.pt        # Trained model weights
├── splits_densenet121.json    # Train/Val/Test split definition
├── requirements.txt           # Python dependencies
└── README.md

## ⚙️ Installation
pip install -r requirements.txt

## ▶️ Inference
A simple inference script will be provided to run prediction on a single fundus image.

## ⚠️ Disclaimer
This project is intended for **research and educational purposes only**.
It is **not a clinically approved diagnostic tool**.
Further validation on larger and more diverse clinical datasets is required.
