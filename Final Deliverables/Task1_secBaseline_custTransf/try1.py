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
import torch.nn.functional as F


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


nltk.download('punkt_tab')

# Advanced Text Preprocessing for multilingual support
def advanced_normalize_text(text, language='en'):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    
    # Language-specific processing
    if language == 'en':
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # For English, remove non-alphabetic chars
    else:
        text = re.sub(r'[^\w\s]', '', text)  # For other languages, keep word characters
        
    text = ' '.join(text.split())  # Remove extra whitespaces
    return text

# Advanced Tokenization with language-specific stopwords
def advanced_tokenize(text, language='en'):
    try:
        if language == 'en':
            stop_words = set(stopwords.words('english'))
        elif language == 'hi':
            # Hindi stopwords - add more as needed
            stop_words = {'का', 'एक', 'में', 'की', 'है', 'यह', 'और', 'से', 'हैं', 'को', 'पर', 'इस', 'होता', 'कि', 'जो', 'कर', 'मे', 'गया', 'करने', 'किया'}
        elif language == 'ta':
            # Tamil stopwords - add more as needed
            stop_words = {'அது', 'இது', 'என்று', 'மற்றும்', 'இந்த', 'அந்த', 'ஒரு', 'என', 'மேலும்', 'அவர்', 'என்ற', 'என்ன', 'போன்ற'}
        else:
            stop_words = set()
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english')) if language == 'en' else set()
       
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]  # Remove stopwords and short tokens
    return ' '.join(tokens)

# Improved Label Generation with Robust Voting
def generate_advanced_label(row):
    values = row[3:]  # Skip 'index', 'text', and 'key' columns
    valid_votes = [str(v) for v in values if str(v) in ["1.0", "0.0", "1", "0"]]  # Convert to string and filter valid votes
    
    if not valid_votes:  # Ignore rows with no valid votes
        return None
    
    # Majority voting 
    count_1 = valid_votes.count("1.0") + valid_votes.count("1")
    count_0 = valid_votes.count("0.0") + valid_votes.count("0")
    
    # More sophisticated voting threshold
    if count_1 > len(valid_votes) * 0.6:  # 60% threshold for majority
        return 1
    elif count_0 > len(valid_votes) * 0.6:
        return 0
    else:
        return None  # Uncertain cases where we can't decide

# Custom dataset class for creating text datasets
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            return torch.tensor(self.texts[idx], dtype=torch.long)

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class RestrictedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(RestrictedSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Use separate linear projections instead of MultiheadAttention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize with small values
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
    
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Create padding mask if provided
        if key_padding_mask is not None:
            # Expand mask for multi-head attention: [batch_size, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        # Create attention mask for restricted context window
        window_mask = torch.ones(seq_len, seq_len, device=query.device)
        
        # Apply sliding window to allow each position to attend to nearby positions
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            window_mask[i, start:end] = 0
        
        # Convert to boolean mask (True means masked positions)
        window_mask = window_mask.bool().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply projections separately
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute scaled dot-product attention
        # Scale factor to prevent exploding values
        scale_factor = float(self.head_dim) ** -0.5
        
        # Batch matrix multiplication: [batch, heads, seq_len, head_dim] x [batch, heads, head_dim, seq_len]
        # -> [batch, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale_factor
        
        # Apply window mask (True means masked positions)
        scores = scores.masked_fill(window_mask, -1e4)
        
        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        
        # Check for NaN in scores and replace with small negative number
        scores = torch.where(torch.isnan(scores), torch.full_like(scores, -1e4), scores)
        
        # Apply softmax to get attention weights
        # Add a small epsilon to prevent numerical issues
        attn_weights = F.softmax(scores, dim=-1) + 1e-6
        
        # Check for NaN in weights and replace with uniform distribution
        if torch.isnan(attn_weights).any():
            print("NaN detected in attention weights! Replacing with uniform distribution.")
            attn_weights = torch.where(
                torch.isnan(attn_weights), 
                torch.ones_like(attn_weights) / seq_len,
                attn_weights
            )
        
        # Weighted sum: [batch, heads, seq_len, seq_len] x [batch, heads, seq_len, head_dim]
        # -> [batch, heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        # One final NaN check
        if torch.isnan(output).any():
            print("NaN detected in attention output! Replacing with original values.")
            output = torch.where(torch.isnan(output), value, output)
        
        return output, attn_weights
    
# Advanced Transformer Model with Restricted Self-Attention
class TransformerWithRestrictedAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, max_len, num_classes=2,
                 n_heads=8, num_encoder_layers=4, dim_feedforward=1024, dropout=0.3, window_size=5):
        super(TransformerWithRestrictedAttention, self).__init__()
        
        # Ensure n_heads divides embedding_dim
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        
        # Embedding layer with pre-trained embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if embedding_matrix is not None:
            # Clip embedding matrix values to prevent extreme values
            embedding_matrix = np.clip(embedding_matrix, -3.0, 3.0)
            # Check for NaN values
            embedding_matrix = np.nan_to_num(embedding_matrix)
            # Convert to tensor and set as weights
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
            
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # LayerNorm for initial embeddings
        self.initial_norm = nn.LayerNorm(embedding_dim)
        
        # Create transformer encoder layers with restricted self-attention
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': RestrictedSelfAttention(embedding_dim, n_heads, window_size),
                'norm1': nn.LayerNorm(embedding_dim),
                'feedforward': nn.Sequential(
                    nn.Linear(embedding_dim, dim_feedforward),
                    nn.GELU(),  # GELU is more stable than ReLU
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embedding_dim)
                ),
                'norm2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_encoder_layers)
        ])
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm for better stability
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Create padding mask (1 for pad tokens, 0 for non-pad)
        padding_mask = (x == 0).to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Check for NaN in embeddings
        if torch.isnan(x).any():
            print("NaN detected in embeddings! Replacing with zeros.")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply dropout and initial normalization
        x = self.embedding_dropout(x)
        x = self.initial_norm(x)
        
        # Apply transformer layers with restricted self-attention
        for layer_idx, layer in enumerate(self.transformer_layers):
            # Self-attention with careful residual connection
            residual = x
            attn_output, _ = layer['attention'](x, x, x, key_padding_mask=padding_mask)
            
            # Check for NaNs after attention
            if torch.isnan(attn_output).any():
                print(f"NaN detected in attention output (layer {layer_idx})! Using original values.")
                attn_output = residual
            
            # Apply dropout to attention output
            attn_output = layer['dropout'](attn_output)
            
            # First residual connection (pre-norm architecture)
            x = layer['norm1'](residual + attn_output)
            
            # Feed forward with careful residual connection
            residual = x
            ff_output = layer['feedforward'](x)
            
            # Check for NaNs after feed-forward
            if torch.isnan(ff_output).any():
                print(f"NaN detected in feedforward output (layer {layer_idx})! Using original values.")
                ff_output = residual
            
            # Apply dropout to feed-forward output
            ff_output = layer['dropout'](ff_output)
            
            # Second residual connection
            x = layer['norm2'](residual + ff_output)
        
        # Global pooling: mean of non-padded tokens
        mask = (~padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        
        # Avoid division by zero
        sum_mask = mask.sum(dim=1)
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
        
        x = (x * mask).sum(dim=1) / sum_mask  # [batch_size, embedding_dim]
        
        # Check for NaNs before classification head
        if torch.isnan(x).any():
            print("NaN detected before classification head! Replacing with zeros.")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Classification head
        x = self.fc_layers(x)
        
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
        
        if np.isnan(val_loss):
            print("NaN validation loss detected! Not saving checkpoint.")
            return
        
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

# Function to prepare mBERT embeddings for texts
def prepare_mbert_embeddings(texts, max_features, max_len, batch_size=16):
    print("Initializing mBERT tokenizer and model...")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
   
    # Move bert model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()
    
    # Create word index and embedding matrix
    word_index = {'[PAD]': 0}  # Reserve 0 for padding
    embedding_dim = 768  # mBERT hidden size
    embeddings_dict = {}
    
    # Process texts in batches to avoid memory issues
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Processing {len(texts)} texts to extract mBERT embeddings...")
    for batch_idx in range(num_batches):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Tokenize batch
            encoded_batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            
            # Move tensors to device
            input_ids = encoded_batch['input_ids'].to(device)
            attention_mask = encoded_batch['attention_mask'].to(device)
            
            # Get mBERT embeddings
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state
                
                # For each text in the batch
                for i in range(len(batch_texts)):
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                    
                    # For each token in the text
                    for j, token in enumerate(tokens):
                        if token == '[PAD]':
                            continue
                            
                        if token not in word_index:
                            word_index[token] = len(word_index)
                        
                        # Store token embedding
                        token_idx = word_index[token]
                        if token_idx < max_features:
                            # Get token embedding from last hidden state
                            token_embedding = last_hidden_states[i, j].cpu().numpy()
                            
                            # Check for NaN or inf values
                            if np.isnan(token_embedding).any() or np.isinf(token_embedding).any():
                                continue
                                
                            # Clip extreme values
                            token_embedding = np.clip(token_embedding, -5.0, 5.0)
                            embeddings_dict[token] = token_embedding
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"Processed batch {batch_idx + 1}/{num_batches}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print("Skipping batch and continuing...")
            continue
    
    # Create embedding matrix
    num_words = min(max_features, len(word_index))
    
    # Initialize with small random values
    np.random.seed(42)  # For reproducibility
    embedding_matrix = np.random.normal(0, 0.05, (num_words, embedding_dim))
    
    # Ensure padding token has all zeros
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    # Fill embedding matrix with token embeddings
    for word, idx in word_index.items():
        if idx < max_features:
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
    
    # Clip values to prevent extremes
    embedding_matrix = np.clip(embedding_matrix, -3.0, 3.0)
    
    # Check for NaN values
    if np.isnan(embedding_matrix).any():
        print("Warning: NaN values detected in embedding matrix. Replacing with small random values...")
        nan_indices = np.where(np.isnan(embedding_matrix))
        for i, j in zip(*nan_indices):
            embedding_matrix[i, j] = np.random.normal(0, 0.01)
    
    # Validate the embedding matrix
    print(f"Embedding matrix stats: min={embedding_matrix.min()}, max={embedding_matrix.max()}, mean={embedding_matrix.mean()}")
    
    # Create sequences
    sequences = []
    for text in texts:
        # Tokenize text using mBERT tokenizer
        tokens = tokenizer.tokenize(text)[:max_len-2]  # -2 for special tokens
        
        # Convert tokens to indices
        seq = []
        for token in tokens:
            if token in word_index and word_index[token] < max_features:
                seq.append(word_index[token])
        
        if not seq:  # Handle empty sequences
            seq = [0]  # Use padding token
        
        sequences.append(seq)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            # Truncate
            new_seq = seq[:max_len]
        else:
            # Pad
            new_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(new_seq)
    
    return np.array(padded_sequences), word_index, embedding_matrix



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
        skipped_batches = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            try:
                texts, labels = texts.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(texts)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    print(f"NaN detected in outputs at batch {batch_idx}! Skipping batch.")
                    skipped_batches += 1
                    continue
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN/Inf loss detected at batch {batch_idx}! Value: {loss.item()}")
                    print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}")
                    print(f"Labels min: {labels.min().item()}, max: {labels.max().item()}")
                    print("Skipping batch.")
                    skipped_batches += 1
                    continue
                
                # Take mean if loss has multiple values
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"NaN/Inf gradient detected in {name}! Skipping batch.")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    skipped_batches += 1
                    continue
                
                # Gradient clipping (increased max_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Record metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, label_indices = torch.max(labels.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == label_indices).sum().item()
                
                # Print progress every 20 batches
                if (batch_idx + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Skipped: {skipped_batches}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                skipped_batches += 1
                continue
        
        # Skip epoch if too many batches were skipped
        if skipped_batches > len(train_loader) * 0.5:
            print(f"Too many batches skipped in epoch {epoch+1}! Moving to next epoch.")
            continue
            
        train_loss = train_loss / (len(train_loader) - skipped_batches) if (len(train_loader) - skipped_batches) > 0 else float('inf')
        train_acc = correct_train / total_train if total_train > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        val_skipped = 0
        
        with torch.no_grad():
            for batch_idx, (texts, labels) in enumerate(val_loader):
                try:
                    texts, labels = texts.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(texts)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print(f"NaN detected in validation outputs at batch {batch_idx}! Skipping batch.")
                        val_skipped += 1
                        continue
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Check for NaN loss
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"NaN/Inf validation loss detected at batch {batch_idx}! Skipping batch.")
                        val_skipped += 1
                        continue
                    
                    # Take mean if loss has multiple values
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    # Record metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, label_indices = torch.max(labels.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == label_indices).sum().item()
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    val_skipped += 1
                    continue
            
        val_loss = val_loss / (len(val_loader) - val_skipped) if (len(val_loader) - val_skipped) > 0 else float('inf')
        val_acc = correct_val / total_val if total_val > 0 else 0
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
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
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Skipped batches: {skipped_batches}/{len(train_loader)}')
    
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
    max_features = 15000  # vocabulary size - increased for multilingual support
    embed_size = 768  # mBERT embedding dimension
    batch_size = 32
    epochs = 15
    learning_rate = 1e-5
    
    # Load data from multiple files
    file_paths = {
        'en': 'train_en_l1.csv',
        'hi': 'train_hi_l1.csv',
        'ta': 'train_ta_l1.csv'
    }
    
    all_data = []
    
    for lang, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            # Add language column for tracking
            df['language'] = lang
            all_data.append(df)
            print(f"Loaded {lang} data: {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: File {path} not found. Skipping.")
    
    if not all_data:
        print("No data files found. Exiting.")
        return
        
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)} rows")
    
    # Advanced preprocessing with language-specific handling
    combined_df['text'] = combined_df.apply(
        lambda row: advanced_normalize_text(row['text'], row['language']), 
        axis=1
    )
    
    combined_df['text'] = combined_df.apply(
        lambda row: advanced_tokenize(row['text'], row['language']), 
        axis=1
    )
    
    # Generate labels with advanced method
    combined_df["label"] = combined_df.apply(generate_advanced_label, axis=1)
    
    # Drop rows with None labels
    combined_df = combined_df.dropna(subset=["label"])
    
    # Get features and labels
    X = list(combined_df['text'])
    y = combined_df['label'].values
    y = y.astype(int)
    
    # Get language information for potential use
    languages = combined_df['language'].values
    
    # One-Hot Encoding for labels
    one_hot_encoder = OneHotEncoder(categories=[np.array([0, 1])], sparse_output=False)
    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))

    # Verify label distribution
    print(f"Label distribution: {np.bincount(y)}")
    print(f"One-hot shape: {y_one_hot.shape}")
    
    # Prepare sequences and mBERT embeddings
    X, word_index, embedding_matrix = prepare_mbert_embeddings(X, max_features, max_len, batch_size=16)
    
    # Stratified K-Fold for better class distribution
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics
    precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
    all_histories = []
    
    # Cross-validation loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/5")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_one_hot[train_index], y_one_hot[val_index]
        
        # Compute class weights for imbalanced data
        class_counts = np.bincount(y.astype(int))
        class_weights = torch.tensor(
            [len(y) / (len(np.unique(y)) * count) for count in class_counts]
        ).float().to(device)

        print(f"Class weights: {class_weights}")
        
        # Create datasets and dataloaders
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,  # Set to 0 for reproducibility
            pin_memory=True if torch.cuda.is_available() else False)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False)
        
        # Initialize Transformer model with restricted self-attention
        model = TransformerWithRestrictedAttention(
            vocab_size=max_features,
            embedding_dim=embed_size,
            embedding_matrix=embedding_matrix,
            max_len=max_len,
            num_classes=2,
            n_heads=6,  # Need to be divisible by embed_size (768)
            num_encoder_layers=2,
            dim_feedforward=768,
            dropout=0.2,
            window_size=10  # Restricted context window size
        ).to(device)
        
        # Loss with class weights for one-hot encoded labels
        # criterion = nn.BCELoss(weight=class_weights)
        criterion = nn.BCEWithLogitsLoss(weight=class_weights, reduction='mean')
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4,  betas=(0.9, 0.999), eps=1e-8)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=8, verbose=True)
        
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
                _, label_indices = torch.max(labels, 1)  # Convert one-hot to indices
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        # Print metrics for this fold
        print(f"\nFold {fold+1} Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        
        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=['Not Hate', 'Hate'],
            output_dict=True
        )
        
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Not Hate', 'Hate']))
        
        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_preds, ['Not Hate', 'Hate'], fold+1)
        
        # Plot training history
        plot_history(history, fold+1)
        
        # Save model for this fold
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'word_index': word_index,
            'one_hot_encoder': one_hot_encoder,
            'config': {
                'max_features': max_features,
                'embed_size': embed_size,
                'max_len': max_len
            }
        }, f'model_fold_{fold+1}.pt')
    
    # Average metrics across all folds
    print("\nAverage Metrics Across All Folds:")
    print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"Precision (Macro): {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall (Macro): {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1-Score (Macro): {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")


def inference():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters - keep the same as training
    max_len = 150
    max_features = 15000
    embed_size = 768
    batch_size = 32
    
    # Load test data from multiple files
    test_file_paths = {
        'en': 'test_en_l1.csv',
        'hi': 'test_hi_l1.csv',
        'ta': 'test_ta_l1.csv'
    }
    
    all_test_data = []
    
    # Function to try different CSV parsing methods
    def load_csv_with_fallbacks(file_path, language):
        print(f"Trying to load {file_path}...")
        
        # List of options to try
        options = [
            # Standard reading
            {"engine": "c", "on_bad_lines": "warn"},
            # Python engine can be more forgiving
            {"engine": "python", "on_bad_lines": "warn"},
            # Try with explicit encoding
            {"engine": "python", "encoding": "utf-8", "on_bad_lines": "warn"},
            # Try with Latin-1 encoding which accepts any byte value
            {"engine": "python", "encoding": "latin1", "on_bad_lines": "warn"},
            # Skip bad lines completely
            {"engine": "python", "encoding": "utf-8", "on_bad_lines": "skip"},
            # Increase field size limit for large cells
            {"engine": "python", "encoding": "utf-8", "on_bad_lines": "skip", "dtype": str}
        ]
        
        # Try each option in sequence
        for i, option in enumerate(options):
            try:
                print(f"  Attempt {i+1}: Using {option}")
                
                # If using the last option, increase the field size limit
                if i == len(options) - 1:
                    import csv
                    import sys
                    max_int = sys.maxsize
                    while True:
                        try:
                            csv.field_size_limit(max_int)
                            break
                        except OverflowError:
                            max_int = int(max_int/10)
                    print(f"  Increased CSV field size limit to {max_int}")
                
                df = pd.read_csv(file_path, **option)
                print(f"  Success! Loaded {len(df)} rows with {df.shape[1]} columns")
                
                # Print sample of data for debugging
                print("\nSample of loaded data:")
                print(df.head(2))
                print("\nColumn names:", df.columns.tolist())
                
                # Add language column
                df['language'] = language
                return df
                
            except Exception as e:
                print(f"  Failed with error: {str(e)}")
                
                # If we're at the last option and still failing, try a more manual approach
                if i == len(options) - 1:
                    print("\nTrying a more manual approach...")
                    try:
                        # Try to read the file line by line
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                        
                        print(f"Read {len(lines)} lines from file")
                        if len(lines) > 1:
                            # Extract header from first line
                            header = lines[0].strip().split(',')
                            print(f"Detected header: {header}")
                            
                            # Build dataframe manually
                            data = []
                            for i, line in enumerate(lines[1:]):
                                try:
                                    # Try to parse each line
                                    values = line.strip().split(',')
                                    if len(values) >= len(header):
                                        # If we have at least as many values as headers, use them
                                        row = values[:len(header)]
                                        data.append(row)
                                except Exception as inner_e:
                                    print(f"  Error on line {i+2}: {str(inner_e)}")
                            
                            print(f"Successfully parsed {len(data)} rows")
                            if data:
                                df = pd.DataFrame(data, columns=header)
                                df['language'] = language
                                return df
                    except Exception as manual_e:
                        print(f"  Manual approach failed: {str(manual_e)}")
        
        print(f"All attempts to read {file_path} failed.")
        return None
    
    # Try to load each file
    for lang, path in test_file_paths.items():
        try:
            df = load_csv_with_fallbacks(path, lang)
            if df is not None:
                all_test_data.append(df)
                print(f"Successfully loaded {lang} data: {len(df)} rows")
            else:
                print(f"Warning: Failed to load {path}. Skipping.")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            print(f"Warning: File {path} could not be processed. Skipping.")
    
    if not all_test_data:
        print("No test data files could be loaded. Exiting.")
        return
        
    # Combine all test datasets
    combined_test_df = pd.concat(all_test_data, ignore_index=True)
    print(f"Combined test dataset size: {len(combined_test_df)} rows")
    print("Combined dataset columns:", combined_test_df.columns.tolist())
    
    # Check if required columns exist
    if 'text' not in combined_test_df.columns:
        # Look for alternative column names
        possible_text_columns = [col for col in combined_test_df.columns if 'text' in col.lower() or 'content' in col.lower() or 'tweet' in col.lower()]
        if possible_text_columns:
            print(f"'text' column not found. Using '{possible_text_columns[0]}' instead.")
            combined_test_df.rename(columns={possible_text_columns[0]: 'text'}, inplace=True)
        else:
            print("Error: No text column found in the data. Please ensure your CSV files have a 'text' column.")
            return
        
    print("Generating labels for test data...")
    
    # Check if vote columns exist before trying to generate labels
    vote_columns = [col for col in combined_test_df.columns 
                   if col not in ['text', 'language', 'original_index']]
    
    if vote_columns:
        try:
            # Apply the label generation function
            combined_test_df['label'] = combined_test_df.apply(generate_advanced_label, axis=1)
            
            # Count labeled samples
            labeled_count = combined_test_df['label'].notnull().sum()
            print(f"Successfully generated labels for {labeled_count} out of {len(combined_test_df)} samples")
            
            # Remove rows with None labels if you want clean evaluation
            labeled_df = combined_test_df.dropna(subset=['label'])
            print(f"Keeping {len(labeled_df)} samples with valid labels for evaluation")
            
            # Convert label to integer
            labeled_df['label'] = labeled_df['label'].astype(int)
            
            # Replace combined_test_df with labeled_df if you want to use only labeled data
            # combined_test_df = labeled_df
            
            # Distribution of generated labels
            print("\nDistribution of generated labels:")
            print(combined_test_df['label'].value_counts())
        except Exception as e:
            print(f"Error generating labels: {e}")
            print("Continuing without labels...")
    else:
        print("No vote columns found. Cannot generate labels for evaluation.")
        
    
    # Apply the same preprocessing as in training
    print("Preprocessing test data...")
    try:
        combined_test_df['text'] = combined_test_df.apply(
            lambda row: advanced_normalize_text(row['text'], row['language']), 
            axis=1
        )
        
        combined_test_df['text'] = combined_test_df.apply(
            lambda row: advanced_tokenize(row['text'], row['language']), 
            axis=1
        )
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Continuing without custom preprocessing...")
    
    # Get test features
    X_test = list(combined_test_df['text'])
    
    # Keep track of the original indices for later
    combined_test_df['original_index'] = combined_test_df.index
    
    # Load the best model checkpoint
    # You can choose which fold's model to use based on validation metrics
    checkpoint_path = 'model_fold_5.pt'  # Change to your best performing model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file {checkpoint_path} not found.")
        all_checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if all_checkpoint_files:
            alternative_checkpoint = all_checkpoint_files[0]
            print(f"Using alternative checkpoint: {alternative_checkpoint}")
            checkpoint = torch.load(alternative_checkpoint, map_location=device)
        else:
            print("No checkpoint files found. Exiting.")
            return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Extract model config and word index
    try:
        word_index = checkpoint['word_index']
        config = checkpoint['config']
        one_hot_encoder = checkpoint.get('one_hot_encoder', None)
        print(f"Loaded model configuration: {config}")
        print(f"Word index size: {len(word_index)}")
    except KeyError as e:
        print(f"Error: Missing key in checkpoint: {e}")
        print("Available keys in checkpoint:", list(checkpoint.keys()))
        return
    
    # Create sequences using the saved word_index
    print("Creating sequences from test data using saved word index...")
    
    # Use the same tokenizer as in training for consistency
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Process test data
    try:
        sequences = []
        for text in X_test:
            # Handle non-string inputs
            if not isinstance(text, str):
                text = str(text)
            
            # Tokenize text using mBERT tokenizer
            tokens = tokenizer.tokenize(text)[:max_len-2]  # -2 for special tokens
            
            # Convert tokens to indices
            seq = []
            for token in tokens:
                if token in word_index and word_index[token] < max_features:
                    seq.append(word_index[token])
                # If token not in word_index, skip it
            
            if not seq:  # Handle empty sequences
                seq = [0]  # Use padding token
            
            sequences.append(seq)
        
        # Pad sequences
        X_test_processed = []
        for seq in sequences:
            if len(seq) > max_len:
                # Truncate
                new_seq = seq[:max_len]
            else:
                # Pad
                new_seq = seq + [0] * (max_len - len(seq))
            X_test_processed.append(new_seq)
        
        X_test_processed = np.array(X_test_processed)
        print(f"Test data shape after processing: {X_test_processed.shape}")
    except Exception as e:
        print(f"Error processing test data: {e}")
        return
    
    # Create test dataset without labels (for inference)
    test_dataset = TextDataset(X_test_processed, np.zeros((len(X_test_processed), 2)))  # Dummy labels
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Important: don't shuffle for inference
        num_workers=0
    )
    
    # Initialize model with the same architecture
    # Replace this part in the inference code
    try:
        # Get the actual vocabulary size from the checkpoint
        actual_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
        print(f"Actual vocabulary size from checkpoint: {actual_vocab_size}")
        
        # Use the actual vocabulary size from the checkpoint instead of max_features
        model = TransformerWithRestrictedAttention(
            vocab_size=actual_vocab_size,  # Use this instead of config['max_features']
            embedding_dim=config['embed_size'],
            embedding_matrix=None,
            max_len=config['max_len'],
            num_classes=2,
            n_heads=6,
            num_encoder_layers=2,
            dim_feedforward=768,
            dropout=0.0,
            window_size=10
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Run inference
    print("Running inference...")
    all_preds = []
    all_probs = []
    
    try:
        with torch.no_grad():
            for texts, _ in tqdm(test_loader, desc="Running inference"):
                texts = texts.to(device)
                outputs = model(texts)
                probabilities = torch.sigmoid(outputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        print(f"Inference complete. Generated {len(all_preds)} predictions.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Add predictions to the test dataframe
    combined_test_df['predicted_label'] = all_preds
    combined_test_df['hate_probability'] = [prob[1] for prob in all_probs]  # Probability of hate class
    
    # Map predictions to human-readable labels
    combined_test_df['prediction'] = combined_test_df['predicted_label'].map({0: 'Not Hate', 1: 'Hate'})
    
    # Print distribution of predictions
    print("\nPredicted Label Distribution:")
    print(combined_test_df['prediction'].value_counts())
    
    # Print distribution by language
    print("\nPrediction Distribution by Language:")
    for lang in combined_test_df['language'].unique():
        lang_df = combined_test_df[combined_test_df['language'] == lang]
        print(f"\n{lang.upper()} Predictions:")
        print(lang_df['prediction'].value_counts())
        print(f"Total {lang} samples: {len(lang_df)}")
    
    # Save predictions to CSV
    try:
        combined_test_df.to_csv('test_predictions.csv', index=False)
        print("\nSaved predictions to test_predictions.csv")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        
    
    # Use the generated labels in your evaluation section
    if 'label' in combined_test_df.columns and combined_test_df['label'].notnull().sum() > 0:
        print("\nEvaluating performance on test set...")
        try:
            # Filter to only use rows with valid labels
            eval_df = combined_test_df.dropna(subset=['label'])
            y_true = eval_df['label'].values
            y_pred = [all_preds[i] for i in eval_df.index]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision (Macro): {precision:.4f}")
            print(f"Test Recall (Macro): {recall:.4f}")
            print(f"Test F1-Score (Macro): {f1:.4f}")
            
            # Detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=['Not Hate', 'Hate']))
            
            # Plot confusion matrix
            plot_confusion_matrix(y_true, y_pred, ['Not Hate', 'Hate'], 'Test Set')
        except Exception as e:
            print(f"Error evaluating performance: {e}")
            
    else:
        print("No valid labels available for evaluation.")

    return combined_test_df


# if __name__ == "__main__":
#     main()

inference()