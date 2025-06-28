# Task 2: Custom Transformer with Restricted Self-Attention Second Baseline
 
## Overview
A PyTorch-based hate speech detection model built using a custom Transformer architecture with a restricted self-attention mechanism. Designed for multilingual text classification tasks such as gendered abuse and toxic language detection, the model leverages contextual embeddings from mBERT along with local attention to focus on relevant token interactions.

## Features
- Multilingual support using bert-base-multilingual-cased
- Combines BERT embeddings with a bidirectional GRU
- Custom restricted self-attention with sliding window
- Final classification through a two-layer feedforward network
- Supports masking and padding
- Modular and easily extendable architecture

## Requirements
```
torch
pandas
numpy
nltk
scikit-learn
transformers
matplotlib
tqdm
```
## Usage
1. Prepare your multilingual data in CSV format
2. Train the model for pretraining with `python try1_trainingHate.py`
3. Train the model for finetuning with `python try1b.py`
4. The model saves checkpoints for in pretrained train data as `final_model.pt`
5. Checkpoint for fine tuned model is saved as `finetuned_gendered_abuse_model.pt`

## Model Architecture
- BERT Embeddings: Uses frozen bert-base-multilingual-cased for extracting contextual token embeddings.
- Multilingual support using bert-base-multilingual-cased
- Custom Transformer encoder with windowed self-attention for local context modeling
- Supports variable sequence lengths with padding and masking

## Evaluation Metrics
- On Hate Dataset:
- - F1 Score (pretraining): 0.7785
- - False Negative (pretraining): 2262
- On ULI dataset:
- - F1 Score (fine-tuning): 0.6094
- - False Negative (fine-tuning): 467

The model automatically handles class imbalance and provides language-specific performance metrics.
