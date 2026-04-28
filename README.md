# Zoidberg 2.0 – Pneumonia Detection AI

Machine learning project for detecting pneumonia from chest X-ray images using classical ML and a bonus CNN implementation.

## Overview

Zoidberg analyzes chest X-ray images and classifies them as either NORMAL or PNEUMONIA using multiple machine learning approaches, including classical ML with PCA and a Convolutional Neural Network.

## Results

| Model                | Val Accuracy | Test Accuracy |
| -------------------- | ------------ | ------------- |
| Logistic Regression  | 94%          | -             |
| Random Forest        | 91%          | -             |
| SVM (best classical) | 97%          | 76%           |
| CNN (bonus)          | 97.8%        | 78%           |

## Pipeline

1. Image loading and grayscale conversion
2. Resizing to 64x64 and normalization (÷255)
3. Flattening to 1D arrays
4. PCA dimensionality reduction (4096 → 100 features)
5. Train/validation/test split (80/20)
6. Model training and hyperparameter tuning with GridSearchCV
7. Evaluation with confusion matrix, ROC-AUC, cross validation

## Tech Stack

- Python
- Jupyter Notebook
- scikit-learn
- TensorFlow / Keras
- PIL
- NumPy
- Matplotlib

## Machine Learning Models

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM) ← best classical model
- Convolutional Neural Network (CNN) ← bonus

## Dataset Structure

```text
chest_Xray/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```

## Key Findings

- SVM achieved best classical ML performance at 97% validation accuracy
- Distribution shift between train and test folders causes drop to 76% on test
- CNN improved test accuracy to 78% by learning deeper spatial features
- Class imbalance (1342 normal vs 3876 pneumonia) handled with class_weight='balanced'
