import pandas as pd
import numpy as np
import re
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data_path = './drive/MyDrive/nlp_dataset/train'
files = ['train_en_l1.csv', 'train_hi_l1.csv', 'train_ta_l1.csv']

# Function to load datasets with language info
def load_data(files, data_path):
    dfs = []
    for file in files:
        lang = file.split('_')[1] 
        df = pd.read_csv(os.path.join(data_path, file))
        df['language'] = lang
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

#Dataset Loading
df = load_data(files, data_path)

# Step 1: Handle Missing Values
def clean_missing_values(df):
    df.replace('NL', np.nan, inplace=True)

    # Drop rows where all annotations are missing
    annotator_cols = [col for col in df.columns if re.match(r".*a[1-6]", col)]
    df.dropna(subset=annotator_cols, how='all', inplace=True)

    return df, annotator_cols

df, annotator_cols = clean_missing_values(df)

# Step 2: Creating Label column using mean
def majority_vote(row):
    votes = row[annotator_cols].dropna().values.astype(float)
    if len(votes) == 0:
        return np.nan
    return 1.0 if votes.mean() >= 0.5 else 0.0

df['label'] = df.apply(majority_vote, axis=1)

# Drop rows where label is still missing
df.dropna(subset=['label'], inplace=True)

# Text Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase the text

    # Preserve hashtags and mentions
    text = re.sub(r'@\w+', '[USER]', text)  
    text = re.sub(r'http\S+|www\S+', '[URL]', text) 

    # Remove tags <user handle>
    text = re.sub(r'<.*?>', ' ', text)

    text = re.sub(r'[^a-zA-Z0-9#@ऀ-ॿ஀-௿\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Step 4: Finalize Dataset
df_final = df[['cleaned_text', 'label', 'language']]

# Save cleaned dataset
df_final.to_csv('./drive/MyDrive/nlp_dataset/train/cleaned_dataset_l1.csv', index=False)

print("Data preparation complete! Cleaned dataset saved as 'cleaned_dataset_l1.csv'.")
print(df_final.head())


#Loading the pretrained model and Training

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# Paths
DATA_PATH = "./drive/MyDrive/nlp_dataset/train/cleaned_dataset_l1.csv"
MODEL_SAVE_PATH = "./drive/MyDrive/nlp_dataset/mbert_gendered_abuse_model"
LOSS_PLOT_PATH = "./drive/MyDrive/nlp_dataset/loss_plot.png"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

data = pd.read_csv(DATA_PATH)

# Initialize mBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Custom Dataset Class
class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 80/20 Split of Test and Train Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['cleaned_text'].values,
    data['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=data['label'].values  # Ensure balanced classes in both sets
)

train_dataset = AbuseDataset(train_texts, train_labels, tokenizer)
val_dataset = AbuseDataset(val_texts, val_labels, tokenizer)

# Load mBERT model (with dropout for better generalization)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2, hidden_dropout_prob=0.3)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./drive/MyDrive/nlp_dataset/results/",
    evaluation_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    report_to="none",  # Disable Weights & Biases
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,  # Apply L2 regularization for better generalization
    load_best_model_at_end=True,  # Save the best model based on eval loss
    metric_for_best_model="eval_loss"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Training the model
train_result = trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Extract loss values for visualization
train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
val_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]

epochs = list(range(1, len(val_losses) + 1))

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(1, len(epochs), len(train_losses)), train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(LOSS_PLOT_PATH)
plt.show()

print(f"Loss plot saved to: {LOSS_PLOT_PATH}")

# Preparing the test dataset

# Paths to test datasets (adjust if needed)
test_data_path = './drive/MyDrive/nlp_dataset/test'
test_files = ['test_en_l1.csv', 'test_ta_l1.csv']

# Function to load datasets with language info
def load_data(files, data_path):
    dfs = []
    for file in files:
        lang = file.split('_')[1]  # Extract language (en, hi, ta)
        df = pd.read_csv(os.path.join(data_path, file))
        df['language'] = lang
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Load all test datasets
test_df = load_data(test_files, test_data_path)

def clean_missing_values(df):
    # Replace 'NL' with np.nan for easier handling
    df.replace('NL', np.nan, inplace=True)

    # Identify annotation columns (en_a1, en_a2, etc.)
    annotator_cols = [col for col in df.columns if re.match(r".*a[1-6]", col)]

    # Drop rows where all annotations are missing
    df.dropna(subset=annotator_cols, how='all', inplace=True)

    return df, annotator_cols

test_df, annotator_cols = clean_missing_values(test_df)

# Step 2: Create Final Label (Majority Vote)
def majority_vote(row):
    # Collect valid annotations (drop NaN values)
    votes = row[annotator_cols].dropna().values.astype(float)

    # If no valid annotations, return NaN
    if len(votes) == 0:
        return np.nan

    # Majority vote (if mean >= 0.5, label = 1, else label = 0)
    return 1.0 if votes.mean() >= 0.5 else 0.0

test_df['label'] = test_df.apply(majority_vote, axis=1)

test_df.dropna(subset=['label'], inplace=True)

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()  # Lowercase the text

    # Preserve hashtags and mentions
    text = re.sub(r'@\w+', '[USER]', text)  # Replace handles with [USER]
    text = re.sub(r'http\S+|www\S+', '[URL]', text)  # Replace URLs with [URL]

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Keep alphanumeric (English, Hindi, Tamil) + hashtags and mentions
    text = re.sub(r'[^a-zA-Z0-9#@ऀ-ॿ஀-௿\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

test_df['cleaned_text'] = test_df['text'].apply(clean_text)

test_df_final = test_df[['cleaned_text', 'label', 'language']]

output_path = './drive/MyDrive/nlp_dataset/test/cleaned_dataset_l1.csv'
test_df_final.to_csv(output_path, index=False)

print("Test data cleaning complete! Cleaned dataset saved as 'cleaned_dataset_l1.csv'.")
print(test_df_final.head())

#Loading the test dataset
import pandas as pd

# Path to your test dataset
TEST_DATA_PATH = "./drive/MyDrive/nlp_dataset/test/cleaned_dataset_l1.csv"

# Load test data
test_data = pd.read_csv(TEST_DATA_PATH)

# Check the structure of the dataset
print(test_data.head())
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_SAVE_PATH = "./drive/MyDrive/nlp_dataset/mbert_gendered_abuse_model"

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)

model.to(device)
model.eval()

#Preparing the test data for evaluation 
from torch.utils.data import Dataset

class TestAbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create the test dataset
test_dataset = TestAbuseDataset(
    test_data["cleaned_text"].values,
    test_data["label"].values,
    tokenizer
)

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)

# Extract logits and convert to predicted labels
predicted_labels = predictions.predictions.argmax(axis=1)

# Compute macro F1 score
macro_f1 = f1_score(test_data["label"], predicted_labels, average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")