# COX-2 Inhibitor Prediction using Machine Learning

This project presents a machine learning framework for predicting COX-2 inhibitory activity using molecular fingerprints and validated through external testing.

---

## 🔬 Overview

- Model: Random Forest (optimized)
- Features: Morgan fingerprints (ECFP4)
- Validation:
  - Internal validation (train/validation split)
  - External validation dataset
  - Y-randomization
  - Applicability Domain (AD) analysis

---

## 🌐 Web Application

The model is deployed as an interactive web application:

👉 https://huggingface.co/spaces/N80-ouass/cox-pred

Users can input SMILES strings to obtain predictions of COX-2 inhibitory activity.

---

## 📊 Dataset

The external validation dataset is provided:
