# Pretrained mBERT Model for Gendered Abuse Detection

This repository implements Gender abuse detection for merged dataset of 3 language(India, English and Tamil) for label 1. mBERT model is used which learn over Gender abuse dataset to classify whether the there is any gender abuse in text.

## Features

- **Text Preprocessing**: Clean missing value, URL removal , Replacing user handles, Placeholder removal 
- **Label Creation**: Derives the majority vote across annotators for each label.
- **Dataset Merging**: Merges datasets by text and assigns corresponding labels.

## Installation

To run this project, install the required dependencies using:

```bash
pip install pandas numpy torch matplotlib scikit-learn transformers 
```

To train the model, run:
```bash
python task1.py
```

Ensure that the Uli dataset with training folder ( containing train_en_l1.csv, train_hi_l1.csv, train_ta_l1.csv) and testing folder ( containing test_en_l1.csv, test_hi_l1.csv, test_ta_l1.csv)

Metrics Recorded after training the model on merged dataset of train_en_l1.csv, train_hi_l1.csv, train_ta_l1.csv.

## Average Metrics:

**Macro F1-Score:** 0.7940

## Outputs
<!-- -**Training history plots** (output_plots/training_history_fold_X.png). -->
-**Trained model checkpoint** (mbert_gendered_abuse_model).

## References
This implementation of baseline is inspired and refered from Hugging face mBERT model 
link: [mBERT ](https://huggingface.co/google-bert/bert-base-multilingual-cased)