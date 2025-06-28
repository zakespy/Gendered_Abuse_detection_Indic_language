# Task 3: Multi-Task Gendered Abuse Detection using GRU-Attention Second Baseline
 
## Overview
This repository contains an implementation of a multi-label classification model to detect gendered abuse and explicit language in social media posts using a hybrid BERT-GRU-Attention architecture.   

## Features
- Supports multi-label classification
- Implements restricted attention for better local context capture
- Includes gradient clipping, learning rate scheduling, and loss plotting
- Modular and ready for extension to other tasks

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
2. Train the model with `nlp-project-task3.ipynb`
3. The model saves checkpoints for train data as label_1 and label_3 classifcation `multiclass_model.pth`

## Model Architecture
- BERT Encoder: Extracts contextual embeddings using bert-base-multilingual-cased.
- GRU Layer: Captures sequential information from BERT embeddings.
- Restricted Self-Attention: Applies localized attention within a fixed window size.
- Classification Head: Predicts multi-class labels (label_1, label_3).

## Evaluation Metrics
- Class: Label 1
- - F1 Score: 0.6008
- - False Negative: 331
- Class: Label 3
- - F1 Score: 0.7328
- - False Negative: 586

The model automatically handles class imbalance and provides language-specific performance metrics.
