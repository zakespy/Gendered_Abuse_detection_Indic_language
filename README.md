# Gendered Abuse Detection in Indic Languages

**Authors:** Kushal Mitra (MT24050), Sankaranarayanan Sengunther (MT24081), Satyam Singh (MT24082)

## 📖 Overview

This repository contains our implementation for the **ICON 2023 Shared Task on Gendered Abuse Detection in Indic Languages**, led by Tattle Civic Tech. The project focuses on detecting gendered abuse in online text across three languages: **Hindi**, **Tamil**, and **Indian English**.

The shared task addresses the critical need to identify and mitigate gender-based harassment and abuse in online spaces, particularly in the Indian digital ecosystem. Our approach implements two novel deep learning architectures to tackle this challenging multilingual problem.

## 🎯 Problem Statement

Online platforms have become fertile ground for gender-based harassment, often manifesting as abusive and exclusionary language. This digital abuse:
- Reinforces systemic inequality and discrimination
- Causes psychological harm and mental health challenges
- Undermines freedom of expression, particularly for marginalized communities
- Creates hostile environments that exclude participation

The complexity increases in Indic language contexts due to:
- Limited digital resources for Indian languages
- Code-switching and hybrid expressions
- Cultural nuances embedded in gendered insults
- Informal, contextual language patterns on social media

The task is part of the ICON 2023 conference shared task initiative, based on a novel dataset curated by activists and researchers with experience in gender violence detection.

## 📊 Dataset Description

Our dataset combines multiple sources to create a comprehensive training corpus:

### Primary Dataset (ICON 2023 Shared Task)
- **English:** 6,532 posts
- **Hindi:** 6,198 posts  
- **Tamil:** 6,780 posts
- **Total:** 19,510 annotated posts
- **Annotators:** 18 researchers and activists from marginalized gender and LGBTQIA+ communities
- **Annotation Approach:** Majority voting with lived experience insights

### Annotation Schema
Each post is labeled across three dimensions:
- **Label 1:** General gendered abuse detection
- **Label 2:** Abuse targeting marginalized groups
- **Label 3:** Explicit or aggressive content classification

Labels include: "1" (yes), "0" (no), "NL" (not labeled), "NaN" (not assigned)

### Auxiliary Datasets
- **MACD Dataset:** Multi-lingual abusive content in 5 Indian languages
- **English Hate Speech Dataset:** Broader linguistic diversity for transfer learning

## 🔬 Experimental Tasks

Our implementation addresses the three subtasks defined in the ICON 2023 shared task:

### Task 1: General Gendered Abuse Detection
Binary classification to identify gendered abuse in posts, regardless of specific targeting.

### Task 2: Transfer Learning for Abusive Content Detection
Leveraging pre-trained models on external hate speech datasets, then fine-tuning on our multilingual corpus.

### Task 3: Multi-label Classification
Simultaneous detection of multiple abuse categories (Label 1 and Label 3).

Each task presents unique challenges in handling code-mixed content, cultural nuances, and language-specific abuse patterns.

## 🏗️ Architecture Overview

We implemented two distinct deep learning architectures:

### Architecture 1: GRU with Restricted Self-Attention

```
Input Text → mBERT Embeddings (Frozen) → Bidirectional GRU → 
Windowed Self-Attention → Mean Pooling → Dense Layers → Classification
```

**Key Features:**
- Frozen mBERT embeddings for rich contextual representation
- Bidirectional GRU for efficient sequential processing
- Windowed self-attention for local dependency capture
- Reduced computational overhead compared to full transformers

**Performance:**
- Task 2: **0.706 Macro F1** (Best performance)
- Task 3 (Q1): 0.6008 Macro F1
- Task 3 (Q3): 0.7328 Macro F1

### Architecture 2: Transformer with Gated Convolutional Network (GCN)

```
Input Text → mBERT Embeddings (Frozen) → Gated CNN → 
Transformer Encoder → CLS Token/Pooling → Classification
```

**Key Features:**
- Gated convolutional layers for selective feature filtering
- Transformer encoder for global contextual understanding
- GLU-inspired gating mechanism for noise suppression
- Effective local and global feature combination

**Performance:**
- Task 1: **0.7516 Macro F1** (Best performance)
- Task 3 (Q1): 0.6585 Macro F1
- Task 3 (Q3): **0.7803 Macro F1** (Best performance)

## ⚙️ Experimental Setup & Hyperparameters

### Common Configuration
- **Preprocessing:** Text normalization, tokenization, and language-specific handling
- **Evaluation:** Stratified cross-validation
- **Metrics:** Precision, Recall, F1-score, and Accuracy

### Architecture-Specific Parameters
**GRU with Restricted Self-Attention:**
- Hidden units: [128, 256, 512]
- Attention heads: [4, 8]
- Dropout: [0.2, 0.3, 0.5]
- Learning rate: [1e-3, 5e-4, 1e-4]

**Transformer with Gated CNN:**
- Transformer layers: [6, 12]
- CNN filters: [64, 128, 256]
- Gate activation: Sigmoid
- Batch size: [16, 32, 64]

## 📈 Evaluation Metrics & Results

### Performance Metrics
- **Precision:** Measures the accuracy of positive predictions
- **Recall:** Measures the ability to identify all positive instances
- **F1-Score:** Harmonic mean of precision and recall
- **Accuracy:** Overall correctness of predictions

### Results Summary
Our models achieved competitive performance across all three languages:

### Test Results Summary

| Task | Architecture | Macro F1 Score | Performance |
|------|-------------|----------------|-------------|
| Task 1 | GCN-Transformer | **0.7516** | Best |
| Task 2 | GRU-Attention | **0.7068** | Best |
| Task 3 (Avg) | GCN-Transformer | **0.7194** | Best |
| Task 3 (Q1) | GCN-Transformer | 0.6585 | - |
| Task 3 (Q3) | GCN-Transformer | **0.7803** | Best |

### Baseline Comparisons

Our models consistently outperformed baseline approaches:
- **mBERT Fine-tuning:** 0.7940 → **0.7516** (Task 1)
- **Custom Transformer:** 0.6994 → **0.7516** (Task 1)
- **CNN-BiLSTM + GloVe:** 0.77 → **0.7516** (Task 1)

*Detailed results and performance comparisons are available in individual task README files.*
<!-- 
## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
torch >= 1.9.0
transformers >= 4.0.0
numpy >= 1.19.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
```

### Installation
```bash
git clone https://github.com/zakespy/Gendered_Abuse_detection_Indic_language.git
cd Gendered_Abuse_detection_Indic_language
pip install -r requirements.txt
```

### Usage
```bash
# For Task 1 - Binary Classification
cd Task1/
python train.py --model gru_attention --epochs 50

# For Task 2 - Multi-class Classification  
cd Task2/
python train.py --model transformer_cnn --batch_size 32

# For Task 3 - Cross-lingual Evaluation
cd Task3/
python cross_lingual_eval.py --source_lang hi --target_lang ta
``` -->

## 🔗 Related Links

- **ICON 2023 Shared Task:** [Official Website](https://sites.google.com/view/icon2023-tattle-sharedtask/overview)
- **Tattle Civic Tech:** [Organization Website](https://tattle.co.in/)
- **Competition Page:** [Kaggle Competition](https://www.kaggle.com/competitions/gendered-abuse-detection-shared-task)

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mitra2024gendered,
  title={Gendered Abuse Detection in Indic Languages},
  author={Mitra, Kushal and Sengunther, Sankaranarayanan and Singh, Satyam},
  year={2024},
  url={https://github.com/zakespy/Gendered_Abuse_detection_Indic_language}
}
```
