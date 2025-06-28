# Pretrained dehatebert Model for Gendered Abuse Detection

This repository implements Transfer Learning Model for Gender abuse detection of merged dataset of 3 label for english dataset. dehatebert learned over hate speech dataset is fine tuned over Gender Abuse Dataset.

## Features
 
- **Label Creation**: Derives the majority vote across annotators for each label.
- **Dataset Merging**: Merges datasets by text and assigns corresponding labels.

## Installation

To run this project, install the required dependencies using:

```bash
pip install pandas numpy torch matplotlib scikit-learn transformers 
```

To train the model, use:
```bash
nlp-project-task2-baseline.ipynb
```

Ensure that the Uli dataset with training (containing train_en_l1.csv, train_hi_l1.csv, train_ta_l1.csv) and testing folders (containing test_en_l1.csv,test_hi_l1.csv and test_ta_l1.csv) are available in the working directory.

Metrics Recorded after training the model on merged dataset of train_en_l1.csv, train_hi_l1.csvand train_ta_l1.csv.

## Average Metrics:

**Macro F1-Score:** 0.6560

## Outputs
<!-- -**Training history plots** (output_plots/training_history_fold_X.png). -->


## References
This implementation of baseline is inspired and refered from Hugging face mBERT model 
link: [MBERT ](https://huggingface.co/google-bert/bert-base-multilingual-cased)