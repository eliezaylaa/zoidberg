# Zoidberg – Pneumonia Detection AI

Machine learning project for detecting pneumonia from chest X-ray images using classical computer vision and scikit-learn models.

## Overview

Zoidberg analyzes chest X-ray images and classifies them as either normal or pneumonia.

The project focuses on image preprocessing, dimensionality reduction, model training, and evaluation using machine learning techniques.

## Features

- Chest X-ray image classification
- Image preprocessing using Python
- Feature extraction and dimensionality reduction with PCA
- Model training with multiple classifiers
- Model comparison and evaluation
- Confusion matrix and classification metrics

## Tech Stack

- Python
- Jupyter Notebook
- scikit-learn
- PIL
- NumPy
- Matplotlib

## Machine Learning Models

The project compares several classification models:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

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
