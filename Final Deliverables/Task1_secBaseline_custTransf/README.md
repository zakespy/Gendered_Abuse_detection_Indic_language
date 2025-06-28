# Task 1: Custom Transformer Second BaseLine with Restricted Self Attention
 
## Overview
A PyTorch implementation of a hate speech detection model using a custom transformer architecture with restricted self-attention. The model supports multiple languages and achieves robust performance through advanced training techniques.

## Features
- Multilingual support (English, Hindi, Tamil)
- Custom transformer with restricted self-attention
- mBERT embedding integration
- Advanced text preprocessing pipeline
- Gradient clipping and early stopping
- Cross-validation with comprehensive metrics

## Requirements
```
torch
pandas
numpy
nltk
scikit-learn
transformers
matplotlib
seaborn
tqdm
```

## Usage
1. Prepare your multilingual data in CSV format
2. Train the model with `python try1.py`
3. The model saves checkpoints for each fold to `model_fold_x.pt`
4. Run inference with `inference()` function
5. Results are saved to `test_predictions.csv`

## Model Architecture
- mBERT embeddings (fine-tuned)
- Positional encoding
- Restricted self-attention mechanism
- Multi-head attention layers
- Classification head with layernorm

## Evaluation Metrics
- F1 Score: 0.6990
- False Negative(FN): 431

The model automatically handles class imbalance and provides language-specific performance metrics.
