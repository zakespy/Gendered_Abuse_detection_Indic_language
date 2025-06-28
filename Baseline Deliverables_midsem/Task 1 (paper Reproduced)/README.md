# Advanced CNN-BiLSTM Model for Gendered Abuse Detection

This repository contains an implementation of an **Advanced CNN-BiLSTM** model for text classification. The model leverages **GloVe embeddings**, **convolutional layers for feature extraction**, and **bi-directional LSTMs for contextual learning**. The training process includes **Stratified K-Fold cross-validation**, **early stopping**, and **learning rate scheduling**.

## Features

- **Text Preprocessing**: Advanced normalization, tokenization, stopword removal, and custom label generation.
- **Word Embeddings**: Uses pre-trained **GloVe embeddings**.
- **Hybrid Model**: CNN layers extract local patterns, and BiLSTM captures long-range dependencies.
- **Regularization**: Dropout and batch normalization for improved generalization.
- **Early Stopping**: Prevents overfitting by monitoring validation loss.
- **Cross-Validation**: Uses **Stratified K-Fold** for better class balance in training.

## Installation

To run this project, install the required dependencies using:

```bash
pip install pandas numpy torch torchvision torchaudio matplotlib seaborn nltk scikit-learn transformers 
```

Additionally, download the required NLTK datasets:

```bash
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

To train the model, run:
```bash
python task1.py
```

Ensure that the dataset (train_en_l1.csv) and GloVe embeddings (glove.6B.300d.txt) are available in the working directory.

Metrics Recorded after training the model on train_en_l1.csv.

## Average Metrics:

**Precision:** 0.8102
**Recall:** 0.7609
**F1-Score:** 0.7769

## Outputs
-**Training history plots** (output_plots/training_history_fold_X.png).
-**Confusion matrices** (output_plots/confusion_matrix_fold_X.png).
-**Trained model checkpoint** (checkpoint.pt).

## References
This implementation is inspired by the CNLP-NITS-PP repository, with modifications for better performance.
link: [CNLP-NITS-PP repository](https://github.com/advaithavetagiri/CNLP-NITS-PP/blob/main/ICON%202023%20Task-3/English/CNN_BiLSTM_Task_3_English_Final.ipynb)
