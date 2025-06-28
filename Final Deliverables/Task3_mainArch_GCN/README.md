# Task 3: Multi-Task Gendered Abuse Detection using GCN-Transformer Main Architecture
 
## Overview
This project implements a multi-label classification model for detecting gendered abuse across multiple categories (e.g., label_1, label_3) using a hybrid GCN + Transformer architecture built on top of a multilingual BERT model.

## Features
- Uses bert-base-multilingual-cased for multilingual support (English, Hindi, Tamil, etc.)
- Combines CNN and Transformer encoders for richer feature learning.
- Supports multi-label classification via BCEWithLogitsLoss.
- Includes detailed training, validation, and testing pipelines.
- Automatically plots and saves training/validation loss curves.
- Outputs test predictions and metrics including confusion matrices and F1 scores.

## Requirements
```
torch
transformers
scikit-learn
numpy
pandas
matplotlib
tqdm
```
## Usage
1. Prepare your multilingual data in CSV format
2. Train the model with `python task3.py`
3. The model saves checkpoints for train data as label_1 and label_3 classifcation `multiclass_model.pth`

## Model Architecture
- Backbone: bert-base-multilingual-cased for multilingual contextual embeddings (frozen during training).
- Graph Encoder: A Graph Convolutional Network (GCN) captures structural and relational information between tokens or posts.
- Transformer Layer: Custom Transformer encoder models local and global contextual dependencies.
- Classification Head: A feed-forward network predicts multi-class labels (e.g., Gendered Abuse, Explicit Language).
- Input Support: Handles multilingual text with variable lengths using padding and attention masking.

## Evaluation Metrics
- Class: Label 1
- - F1 Score: 0.6585
- - False Negative: 297 
- Class: Label 3
- - F1 Score: 0.7803
- - False Negative: 454

The model automatically handles class imbalance and provides language-specific performance metrics.
