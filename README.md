# DermaVerseAI-Skin-Lesion-Detection-using-Deep-Learning
U²-Net segmentation + EffecientNetb0 hierarchical classification on ISIC 2018
# Segmentation Module (U²-Net)

This repository contains the skin lesion segmentation component of a hierarchical deep learning pipeline for dermoscopic image analysis. The module performs binary lesion segmentation to separate lesion regions from background skin and artifacts.

The segmentation model is based on U²-NetP, a lightweight nested U-shaped architecture designed for accurate foreground–background separation in medical images.

## Dataset
- ISIC 2018 Skin Lesion Segmentation Dataset
- Dataset is not included in this repository

Expected structure:
data/images  
data/masks  

## Preprocessing
- Image and mask resizing to 256 × 256
- Pixel intensity normalization
- Binary mask preprocessing
- Data augmentation applied during training

## Training
- Supervised learning with image–mask pairs
- Loss: Binary Cross-Entropy + Dice Loss
- Optimizer: Adam
- Train/validation split defined in `splits/`

Run:
python train.py

## Evaluation
- Metrics: Dice Score, Intersection over Union (IoU)

Run:
python eval.py

## Notes
- Datasets, trained weights, and generated outputs are excluded from version control
- This module is intended to be reproducible and independently executable
