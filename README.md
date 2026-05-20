# Zoidberg 2.0 – Pneumonia Detection AI

Machine learning project for detecting pneumonia from chest X-ray images using classical ML and a bonus CNN implementation.

## Live Demo
🫁 [Try the live demo on Hugging Face](https://huggingface.co/spaces/eliezaylaa/zoidberg)

## Notebook HTML Export
📓 [View the full notebook](https://eliezaylaa.github.io/zoidberg/chest_Xray/zoidberg.html)

## Overview
Zoidberg 2.0 analyzes chest X-ray images and classifies them as either NORMAL or PNEUMONIA using multiple machine learning approaches, including classical ML with PCA and a bonus Convolutional Neural Network.

## Results

| Model | Val Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 94% | 78% |
| Random Forest | 92% | 67% |
| SVM (best classical) | 97% | 76% |
| CNN (best overall) | 97% | 79% |

## Pipeline
1. Image loading and grayscale conversion
2. Resizing to 64x64 and normalization (÷255)
3. Flattening to 1D arrays (4096 features)
4. PCA dimensionality reduction (4096 → 100 features)
5. Train/validation split 80/20 (4172 train / 1044 val)
6. Model training with class_weight='balanced' for imbalanced data
7. Hyperparameter tuning with GridSearchCV
8. Evaluation: confusion matrix, ROC-AUC, cross validation, classification report

## Tech Stack
- Python
- Jupyter Notebook
- scikit-learn
- TensorFlow / Keras
- PIL
- NumPy
- Matplotlib

## Machine Learning Models
- **Logistic Regression** — simplest baseline, lightest model
- **Random Forest** — 200 trees, ensemble method
- **Support Vector Machine (SVM)** ← best classical model (C=10, rbf kernel)
- **Convolutional Neural Network (CNN)** ← best overall (Early Stopping + ModelCheckpoint)

## Dataset Structure
chest_Xray/
train/    (5216 images: 1342 NORMAL + 3876 PNEUMONIA)
test/     (624 images — completely hidden during training)
val/      (18 images — too small, ignored)

## Key Findings
- CNN achieved best test accuracy at 79% — outperforms all classical models on unseen data
- SVM achieved best classical ML performance at 97% validation accuracy
- Distribution shift between train and test folders causes accuracy to drop from 97% to 76-79%
- Class imbalance (1342 normal vs 3876 pneumonia) handled with class_weight='balanced'
- Early Stopping + ModelCheckpoint prevented overfitting in CNN training
- AUC = 1.00 for SVM and CNN — perfect class separation on validation set
