import torch 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from dataGen import generate_singleLabel_data, generate_multilabel_data, get_pretrained_data
import csv


class MultiClassAbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # self.texts = [self._preprocess_text(text) for text in texts]
        self.texts = texts  # Assuming texts are already preprocessed
        self.labels = labels  # Now labels is a 2D array/list with shape [n_samples, n_classes]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _preprocess_text(self, text):
        # Basic text cleaning
        if not isinstance(text, str):
            text = str(text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)


# Multi-class GCN Transformer
class MultiClassGCNTransformer(nn.Module):
    def __init__(self, num_classes=2, pretrained_model='bert-base-multilingual-cased', hidden_dim=768, nos_layer=1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.embedding = AutoModel.from_pretrained(pretrained_model).embeddings
        self.num_classes = num_classes
        
        # Freeze the embedding weights to prevent training
        for param in self.embedding.parameters():
            param.requires_grad = False
            
        self.conv_feat = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=nos_layer)
        
        # Output layer for multi-class classification
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        
        # Apply attention mask to embeddings
        embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        x = embeddings.permute(0, 2, 1)  # (batch, hidden, seq_len)
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        gated = feat * gate  # (batch, hidden, seq_len)

        gated = gated.permute(2, 0, 1)  # (seq_len, batch, hidden)
        gated = self.layer_norm(gated)  # Apply layer normalization
        
        # Create a mask for the transformer
        transformer_mask = (1 - attention_mask).bool()  # Invert the attention mask
        encoded = self.encoder(gated, src_key_padding_mask=transformer_mask)  # (seq_len, batch, hidden)
        
        # Use mean pooling with attention mask for better representation
        mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size(1), encoded.size(0), encoded.size(2))
        mask_expanded = mask_expanded.permute(1, 0, 2)
        encoded_masked = encoded * mask_expanded
        sum_embeddings = encoded_masked.sum(dim=0)
        sum_mask = mask_expanded.sum(dim=0) + 1e-9  # Add epsilon to avoid division by zero
        pooled = sum_embeddings / sum_mask
        
        # Multi-class output
        logits = self.classifier(self.dropout(pooled))
        return logits


# Function to process the dataset and extract multi-class labels
def process_multiclass_data(df):
    # df = pd.read_csv(file_path)
    texts = df['text'].astype(str).tolist()
    
    # Extract labels for key_1 and key_2
    labels = df[['label_1', 'label_2']].values.tolist()
    
    # Extract keys for reference
    keys = [df['key_1'].iloc[0], df['key_2'].iloc[0]]
    
    return texts, labels, keys


def train_multiclass(model, train_dataloader, val_dataloader, optimizer, device, epochs=3, class_weights=None):
    total_train_loss = []
    total_val_loss = []
    total_steps = len(train_dataloader) * epochs
    
    # Define loss function with class weights if provided
    if class_weights is not None:
        weights = torch.tensor(class_weights).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            # Print learning rate and loss occasionally
            if i % 100 == 0 and i > 0:
                print(f"Batch {i}, Current LR: {scheduler.get_last_lr()[0]:.2e}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)
        total_train_loss.append(avg_train_loss)
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Convert lists to numpy arrays for evaluation
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics for each class
        f1_scores = []
        accuracies = []
        
        for i in range(model.num_classes):
            class_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary')
            class_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
            f1_scores.append(class_f1)
            accuracies.append(class_acc)
        
        avg_val_loss = val_loss / len(val_dataloader)
        total_val_loss.append(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        for i in range(model.num_classes):
            print(f"Class {i+1} - F1: {f1_scores[i]:.4f}, Accuracy: {accuracies[i]:.4f}")
        
        print('-' * 100)

    # Plot the loss graph 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(total_train_loss) + 1), total_train_loss, marker='o', label="Training Loss")
    plt.plot(range(1, len(total_val_loss) + 1), total_val_loss, marker='x', label="Validation Loss", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('multiclass_training_loss.png')
    plt.show()
    
    return total_train_loss, total_val_loss


def test_multiclass(model, test_dataloader, device, keys):
    model.eval()
    all_preds = []
    all_labels = []
    all_texts = []
    
    test_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_texts.extend(batch['input_ids'].cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Print overall test results
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"\nTest Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Save predictions and labels to CSV
    with open('./results/task3_results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ['Text'] + [f'Predicted_{key}' for key in keys] + [f'True_{key}' for key in keys]
        writer.writerow(header)
        
        for text, pred, label in zip(all_texts, all_preds, all_labels):
            decoded_text = tokenizer.decode(text, skip_special_tokens=True)
            row = [decoded_text] + pred.tolist() + label.tolist()
            writer.writerow(row)
    
    print("Test results saved to 'test_results.csv'")
    
    # Calculate and print metrics for each class
    for i in range(model.num_classes):
        class_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary')
        class_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        
        print(f"\nClass: {keys[i]}")
        print(f"F1 Score: {class_f1:.4f}")
        print(f"Accuracy: {class_acc:.4f}")
        
        # Create confusion matrix for each class
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {keys[i]}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add text annotations
        thresh = cm.max() / 2
        for i_cm in range(cm.shape[0]):
            for j_cm in range(cm.shape[1]):
                plt.text(j_cm, i_cm, format(cm[i_cm, j_cm], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i_cm, j_cm] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{keys[i]}.png')
        plt.show()
    
    # Calculate overall metrics (macro average)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\nOverall Macro F1 Score: {macro_f1:.4f}")
    
    return all_preds, all_labels


def calculate_class_weights(labels):
    """Calculate class weights for handling imbalance in multi-class setting"""
    weights = []
    
    # For each class
    for i in range(labels.shape[1]):
        class_labels = labels[:, i]
        pos_count = np.sum(class_labels)
        neg_count = len(class_labels) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            weights.append(1.0)
        else:
            weights.append(neg_count / pos_count)
    
    return weights


# Main function
if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process data
    print("\nLoading and processing multiclass data...")
    
    # Replace with your actual data file path
    # train_file_path = './uli_dataset/training/data.csv'
    # test_file_path = './uli_dataset/testing/data.csv'
    
    train_df = generate_multilabel_data('./uli_dataset/training/', ['l1','l3'])
    
    train_texts, train_labels, keys = process_multiclass_data(train_df)
    print(f"Keys found in dataset: {keys}")
    
    # Convert labels to numpy array for easier handling
    train_labels = np.array(train_labels)
    
    # Split data for training and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=seed
    )
    
    # Calculate class weights for handling imbalance
    class_weights = calculate_class_weights(train_labels)
    print(f"Class weights: {class_weights}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Create datasets
    train_ds = MultiClassAbuseDataset(train_texts, train_labels, tokenizer)
    val_ds = MultiClassAbuseDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Check a sample from dataloader to ensure correct formatting
    sample_batch = next(iter(train_loader))
    print(f"Sample batch - Input IDs shape: {sample_batch['input_ids'].shape}")
    print(f"Sample batch - Labels shape: {sample_batch['labels'].shape}")
    
    # Initialize model for multiclass classification
    num_classes = len(keys)  # Number of keys to classify
    model = MultiClassGCNTransformer(num_classes=num_classes).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01)
    
    # Train model
    print("\nTraining multiclass model...")
    train_multiclass(model, train_loader, val_loader, optimizer, device, epochs=20, class_weights=class_weights)
    
    # Save trained model
    torch.save(model.state_dict(), './multiclass_model.pth')
    print("Multiclass model saved to ./multiclass_model.pth")
    
    
    
    # Test model
    print("\nTesting multiclass model...")
    
    # model.load_state_dict(torch.load('./multiclass_model.pth', map_location=device))
    
    test_df = generate_multilabel_data('./uli_dataset/testing/', ['l1','l3'])
    test_texts, test_labels, test_keys = process_multiclass_data(test_df)
    test_labels = np.array(test_labels)
    
    test_ds = MultiClassAbuseDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    test_multiclass(model, test_loader, device, keys)