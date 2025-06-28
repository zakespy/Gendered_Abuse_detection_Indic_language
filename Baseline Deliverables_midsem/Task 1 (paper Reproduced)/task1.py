import pandas as pd
import numpy as np
import re
import string
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import BertTokenizer  

# Advanced Text Preprocessing (removing irrelevant stuffs and handling casing)
def advanced_normalize_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    text = ' '.join(text.split()) # Remove extra whitespaces  
    return text

# Advanced Tokenization using nltk with Stop Word Removal
def advanced_tokenize(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords') # Download stopwords if not already downloaded
        stop_words = set(stopwords.words('english'))
       
    tokens = nltk.word_tokenize(text) # Tokenize
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2] # Remove stopwords and short tokens
    return ' '.join(tokens)

# Improved Label Generation with More Robust Voting
def generate_advanced_label(row):
    values = row[3:]  # Skip 'index', 'text', and 'key' columns
    valid_votes = [str(v) for v in values if str(v) in ["1.0", "0.0"]]  # Convert to string and filter valid votes
    
    if not valid_votes: # Ignore rows with no valid votes
        return None
    
    # Majority voting 
    count_1 = valid_votes.count("1.0")
    count_0 = valid_votes.count("0.0")
    
    # More sophisticated voting threshold
    if count_1 > len(valid_votes) * 0.6: # 60% threshold for majority
        return 1
    elif count_0 > len(valid_votes) * 0.6:
        return 0
    else:
        return None  # Uncertain cases where we can't decide

# Custom dataset class for creating text datasets to pass to DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.tensor(self.texts[idx], dtype=torch.long)

# Advanced CNN-BiLSTM Model with Improved Dropout and Architecture (The Below Architecture Serves as a baseline model)
# The Architecture is a combination of CNN and BiLSTM with improved dropout techniques and increased conv layers + Uses Adaptive Pooling
# The Model Architecture is refered from the below repository (Tried Reprooducing the results with some modifications)
# https://github.com/advaithavetagiri/CNLP-NITS-PP/blob/main/ICON%202023%20Task-3/English/CNN_BiLSTM_Task_3_English_Final.ipynb

class AdvancedCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, max_len, num_classes=2):
        super(AdvancedCNNBiLSTM, self).__init__()
        
        # Embedding layer with pre-trained embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embedding.weight.requires_grad = False
        
        # Dropout for embedding layer
        self.embedding_dropout = nn.Dropout(0.5)
        
        # Multiple convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, 64, kernel_size=k, padding='same') 
            for k in [2, 3, 4]
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_len) # Adaptive pooling to standardize tensor sizes
        
        self.batch_norm = nn.BatchNorm1d(64 * 3) # Batch normalization for convolutional layers
        
        # Multi-layer BiDirectional LSTM with robust dropout
        self.bilstm = nn.LSTM(
            input_size=64 * 3, 
            hidden_size=128, 
            num_layers=2, 
            bidirectional=True, 
            dropout=0.5, 
            batch_first=True
        )
        
        # Fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),  # 256 because of bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.embedding_dropout(x)
        
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]   # Prepare for convolutional layers
        conv_results = [torch.relu(conv(x)) for conv in self.conv_layers]
        conv_results = [self.adaptive_pool(result) for result in conv_results] # Apply adaptive pooling to standardize sizes
             
        x = torch.cat(conv_results, dim=1) # Concatenate features
        x = self.batch_norm(x) # Batch normalization
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 64*3] # Prepare for LSTM
        
        x, _ = self.bilstm(x) # BiLSTM
        x, _ = torch.max(x, dim=1) # Global max pooling
        x = self.fc_layers(x) # Fully connected layers
        x = self.softmax(x)
        
        return x
    
# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss    

# Function to build vocabulary and prepare sequences
def build_vocab_and_prepare_sequences(texts, max_features, max_len):
    # # Use a basic tokenizer for creating word index (to match with GloVe)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       
    word_index = {} # Create word index for GloVe compatibility
    word_counts = {}
    
    for text in texts: # Process texts to get word counts
        for word in text.split():
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    wcounts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True) # Sort by frequency
    
    for i, (word, _) in enumerate(wcounts): # Build word index (limited to max_features)
        if i >= max_features - 1:  # Save spot for padding token
            break
        word_index[word] = i + 1  # Reserve 0 for padding
    
    # Convert texts to sequences
    sequences = []
    for text in texts:
        seq = []
        for word in text.split():
            if word in word_index:
                seq.append(word_index[word])
        if not seq:  # Handle empty sequences
            seq = [0]
        sequences.append(seq)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            # Truncate
            new_seq = seq[:max_len]
        else:
            # Pad (post)
            new_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(new_seq)
    
    return np.array(padded_sequences), word_index

def load_glove_embeddings(glove_path, word_index, embedding_dim, max_features): # Function to load GloVe embeddings
    embeddings_dictionary = {}
    
    with open(glove_path, encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
    
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, index in word_index.items():
        if index >= max_features:
            continue
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    return embedding_matrix

# Training function with improved monitoring
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, early_stopping):
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()  # Backward and optimize
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        
        scheduler.step(val_loss) # Learning rate scheduling
        
        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

# Plot training history
def plot_history(history, fold_number):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - Fold {fold_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Fold {fold_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'output_plots/training_history_fold_{fold_number}.png')
    plt.close()  # Close the plot to free up memory

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, fold_number):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Fold {fold_number}')
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'output_plots/confusion_matrix_fold_{fold_number}.png')
    plt.close()  # Close the plot to free up memory

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    max_len = 150  # sequence length
    max_features = 10000  # vocabulary size
    embed_size = 300
    batch_size = 64  # batch size
    epochs = 15  # Increased epochs
    learning_rate = 1e-3
    
    # Load data
    file_path = 'train_en_l1.csv'  # Change to your local path
    df = pd.read_csv(file_path)
    
    # Advanced preprocessing
    df['text'] = df['text'].apply(advanced_normalize_text)
    df['text'] = df['text'].apply(advanced_tokenize)
    
    
    df["label"] = df.apply(generate_advanced_label, axis=1) # Generate labels with advanced method
    
    df = df.dropna(subset=["label"]) # Drop rows with None labels
    
    X = list(df['text']) # Get features and labels
    y = df['label'].values
    
    label_encoder = LabelEncoder() # Label encoding !! which is better one hot or label (use whichever is better)
    y = label_encoder.fit_transform(y)
    
    X, word_index = build_vocab_and_prepare_sequences(X, max_features, max_len)  # Tokenize and prepare sequences
    
    # Load GloVe embeddings # (change this to use embeddings of MBERT)
    glove_path = 'glove.6B.300d.txt'  # Change to your local path
    embedding_matrix = load_glove_embeddings(glove_path, word_index, embed_size, max_features)
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Stratified K-Fold for better class distribution
     
    precision_list, recall_list, f1_score_list = [], [], []  # Initialize lists to store metrics
    all_histories = []
     
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)): # Cross-validation loop
        print(f"\nFold {fold+1}/5")
        
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Compute class weights
        class_weights = torch.tensor(
            [len(y_train) / (2 * np.sum(y_train == c)) for c in [0, 1]]
        ).float().to(device)
        
        # Create datasets and dataloaders
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = AdvancedCNNBiLSTM(
            max_features, embed_size, embedding_matrix, max_len
        ).to(device)
        
        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Adaptive learning rate optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
       # Cosine Annealing Scheduler with Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,       # Initial restart epoch
            T_mult=2,    # Multiplicative factor for subsequent restarts
            eta_min=1e-5 # Minimum learning rate
        )

        early_stopping = EarlyStopping(patience=7, verbose=True)
        
        # Train model
        history = train_model(
            model, train_loader, val_loader, 
            criterion, optimizer, scheduler, epochs, device, early_stopping
        )

        all_histories.append(history)
        
        # Evaluate model
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification metrics
        report = classification_report(
            all_labels, all_preds, 
            target_names=['Not Hate', 'Hate'], 
            output_dict=True
        )
        
        precision_list.append(report['weighted avg']['precision'])
        recall_list.append(report['weighted avg']['recall'])
        f1_score_list.append(report['weighted avg']['f1-score'])
        
        
        plot_confusion_matrix(all_labels, all_preds, ['Not Hate', 'Hate'], fold+1) # Plot confusion matrix
    
    # Average metrics
    print("\nAverage Metrics:")
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f}")
    print(f"F1-Score: {np.mean(f1_score_list):.4f}") #calculate macro f1 score instead of this
    
    
    plot_history(all_histories[-1], fold+1) # Plot training history for the last fold

if __name__ == "__main__":
    main() 