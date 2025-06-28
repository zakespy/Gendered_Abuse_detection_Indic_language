# Pretrained IndicBERT Model for Gendered Abuse Detection

This repository implements  Multitask classification model for gender abuse detection. Model predict two binary label (Label 1 and label 3) based on merged dataset

## Features

- **Text Preprocessing**: Removes URLs, special characters (excluding Indic scripts and punctuation), and placeholders.
- **Label Creation**: Derives the majority vote across annotators for each labe.
- **Dataset Merging**: Merges datasets by text and assigns corresponding labels.

## Installation

To run this project, install the required dependencies using:

```bash
pip install pandas numpy torch matplotlib  nltk scikit-learn transformers 
```

To train the model, run:
```bash
python task3.py
```

Ensure that the Uli dataset with training (containing train_en_l1.csv, train_en_l3.csv, train_hi_l1.csv, train_hi_l3.csv, train_ta_l1.csv, train_ta_l3.csv) and testing folders (containing test_en_l1.csv and test_en_l3.csv, test_hi_l1.csv and test_hi_l3.csv, test_ta_l1.csv and test_ta_l3.csv) are available in the working directory.


## Average Metrics:

**Macro F1-Score:** 0.76

## Outputs
<!-- -**Training history plots** (output_plots/training_history_fold_X.png). -->
-**Trained model checkpoint** (task3model.pth).

## References
This implementation of baseline is inspired and refered from Hugging face IndicBERT model 
link: [IndicBERT ](https://huggingface.co/ai4bharat/indic-bert)