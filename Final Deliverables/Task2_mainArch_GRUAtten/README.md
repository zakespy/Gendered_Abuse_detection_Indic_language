# Task 2: GRU with Restricted Self-Attention Main Architecture
 
## Overview
A PyTorch-based hate speech detection model that combines multilingual BERT embeddings with a bidirectional GRU and a restricted self-attention mechanism. This hybrid model is designed to extract both contextual and sequential features effectively, particularly for tasks like gendered abuse or toxic language detection in multilingual datasets.

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
2. Train the model with `nlp-project-task2.ipynb`
3. The model saves checkpoints for in pretrained train data as `best_model_transfer_learning.pt`
4. Checkpoint for fine tuned model is saved as `best_model_fine_tuned.pt`

## Model Architecture
- BERT Embeddings: Uses frozen bert-base-multilingual-cased for extracting contextual token embeddings.
- Bidirectional GRU: Captures sequential dependencies over the BERT embeddings.
- Restricted Self-Attention: Applies attention within a fixed-size window to focus on local context.
- Fully connected layer with ReLU and dropout
- Final linear layer followed by sigmoid activation for binary output

## Evaluation Metrics
- On Hate Dataset:
- - F1 Score (pretraining): 0.8603
- - False Negative (pretraining): 1201
- On ULI dataset:
- - F1 Score (fine-tuning): 0.7068
- - False Negative (fine-tuning): 413

The model automatically handles class imbalance and provides language-specific performance metrics.
