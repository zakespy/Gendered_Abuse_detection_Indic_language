# Task 1: GCN (Gated Convolutional Network) Main Architecture

## Overview
A PyTorch implementation of a text abuse detection model combining BERT embeddings with gated convolutions and transformer architecture. The model is designed for multilingual text classification to identify abusive content.

## Features
- Multilingual support via BERT
- Gated CNN + Transformer architecture
- Automatic threshold optimization
- Comprehensive evaluation metrics

## Requirements
```
torch
numpy
pandas
tqdm
sklearn
transformers
matplotlib
```

## Usage
1. Prepare your data using the `dataGen.py` utility
2. Train the model with `python task1.py`
3. The model will save the best checkpoint to `task1model.pth`
4. Evaluation metrics and visualizations will be generated automatically

## Model Architecture
- BERT embeddings (frozen)
- Gated convolutional layers
- Transformer encoder
- Binary classification head

## Evaluation F1 score
- F1 Score: 0.755
- False Negative(FN): 432

The model automatically determines the optimal classification threshold to maximize F1 score.
