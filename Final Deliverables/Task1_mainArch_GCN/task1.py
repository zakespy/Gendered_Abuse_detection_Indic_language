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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , precision_recall_curve
from dataGen import generate_singleLabel_data, generate_multilabel_data, singleLang_singleLabel



class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)

# 2. Model with Gated Convolution + Transformer
class GCNTransformer(nn.Module):
    def __init__(self, pretrained_model='bert-base-multilingual-cased', hidden_dim=768):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.embedding = AutoModel.from_pretrained(pretrained_model).embeddings
        
        # Freeze the embedding weights to prevent training
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.conv_feat = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=1)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.threshold = 0.5  # Default threshold, will be optimized

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        
        x = embeddings.permute(0, 2, 1)  # (batch, hidden, seq_len)
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        gated = feat * gate  # (batch, hidden, seq_len)

        gated = gated.permute(2, 0, 1)  # (seq_len, batch, hidden)
        encoded = self.encoder(gated)  # (seq_len, batch, hidden)

        pooled = encoded[0]
        # pooled = encoded.mean(dim=0)  # (batch, hidden)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits
    
    def predict(self, input_ids, attention_mask):
        """Use the optimal threshold for prediction"""
        outputs = self(input_ids, attention_mask)
        return torch.sigmoid(outputs) > self.threshold

# Function to find optimal threshold that maximizes macro F1 score
def find_optimal_threshold(model, val_dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Finding optimal threshold"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, mask)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Try different thresholds to find the one that maximizes macro F1
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        preds = (np.array(all_probs) > threshold).astype(int)
        macro_f1 = f1_score(all_labels, preds, average='macro')
        f1_scores.append(macro_f1)
    
    # Find threshold with best macro F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.4f} with Macro F1 score: {best_f1:.4f}")
    
    # precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # # Calculate F1 score for each threshold
    # f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # # Find threshold with the best F1 score
    # best_threshold_idx = np.argmax(f1_scores[:-1])  # Exclude the last element as precision_recall_curve returns len(thresholds)+1 values for precision and recall
    # best_threshold = thresholds[best_threshold_idx]
    # best_f1 = f1_scores[best_threshold_idx]
    
    # print(f"Optimal threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
    # return best_threshold
    
    return best_threshold

# 3. Training and Evaluation
def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=3):
    
    total_train_loss = []
    total_val_loss = []
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        total_train_loss.append(train_loss / len(train_dataloader))
        
        # Find optimal threshold on validation set
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:  # Update threshold every 2 epochs or at last epoch
            model.threshold = find_optimal_threshold(model, val_dataloader, device)
        
        # Evaluate on validation set using the new threshold
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > model.threshold  # Use the optimal threshold
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        f1 = f1_score(all_labels, all_preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            # Save model with threshold
            state_dict = {
                'model_state_dict': model.state_dict(),
                'threshold': model.threshold
            }
            torch.save(state_dict, './task1model.pth')
            
        total_val_loss.append(val_loss / len(val_dataloader))
        print(f"\n Epoch {epoch+1} - Train Loss: {train_loss / len(train_dataloader):.4f}")
        print(f"Epoch {epoch+1} - Val Loss: {val_loss / len(val_dataloader):.4f} - Macro F1: {f1:.4f} - Threshold: {model.threshold:.4f} \n")
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


    
    

    # plotting the loss graph 
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(total_train_loss)), total_train_loss, label="Training Loss")
    plt.plot(range(len(total_val_loss)), total_val_loss, label="Validation Loss", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
def testing(model, test_dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, mask)
            probs = torch.sigmoid(outputs)
            preds = probs > model.threshold  # Use the optimal threshold
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    print("Test Macro F1 Score:", macro_f1)
    print("Test Accuracy Score:", acc)
    print(f"Using threshold: {model.threshold:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (threshold={model.threshold:.4f})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Abuse', 'Abuse'])
    plt.yticks(tick_marks, ['Not Abuse', 'Abuse'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_threshold_{model.threshold:.4f}.png')
    plt.close()
    
    # Plot probability distribution with threshold
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_probs, bins=50, alpha=0.7)
    # plt.axvline(x=model.threshold, color='r', linestyle='--', 
    #             label=f'Optimal Threshold: {model.threshold:.4f}')
    # plt.xlabel('Prediction Probability')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Prediction Probabilities')
    # plt.legend()
    # plt.savefig('prediction_distribution.png')
    # plt.close()
    
    # Additional analysis: Try different thresholds on test set
    test_thresholds = np.arange(0.1, 0.9, 0.05)
    test_f1_scores = []
    
    for threshold in test_thresholds:
        preds = (np.array(all_probs) > threshold).astype(int)
        test_macro_f1 = f1_score(all_labels, preds, average='macro')
        test_f1_scores.append(test_macro_f1)
    
    # Plot F1 scores vs thresholds for test set
    # plt.figure(figsize=(10, 6))
    # plt.plot(test_thresholds, test_f1_scores)
    # plt.axvline(x=model.threshold, color='r', linestyle='--', 
    #             label=f'Current Threshold: {model.threshold:.4f}')
    best_test_threshold = test_thresholds[np.argmax(test_f1_scores)]
    # plt.axvline(x=best_test_threshold, color='g', linestyle='--', 
    #             label=f'Best Test Threshold: {best_test_threshold:.4f}')
    # plt.xlabel('Threshold')
    # plt.ylabel('Test Macro F1 Score')
    # plt.title('Threshold vs Test Macro F1 Score')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('test_threshold_vs_f1.png')
    # plt.close()
    
    # Print both train and test optimal thresholds
    print(f"Validation optimal threshold: {model.threshold:.4f}")
    print(f"Test optimal threshold: {best_test_threshold:.4f}")
    print(f"Test F1 with validation threshold: {macro_f1:.4f}")
    print(f"Test F1 with test optimal threshold: {max(test_f1_scores):.4f}")

# 4. Run Training
if __name__ == "__main__":
    print("Loading Data.......")
    df = generate_singleLabel_data('../uli_dataset/training/', 'l1')
    # df = singleLang_singleLabel('./uli_dataset/training/','en','l1')
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    train_ds = AbuseDataset(train_texts, train_labels, tokenizer)
    val_ds = AbuseDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-9)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Training Model .......")
    train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10)
    
    # Model Loading 
    model = GCNTransformer().to(device)
    checkpoint = torch.load('./task1model.pth', map_location=device,weights_only=False)
    
    # Load model weights and threshold
    model.load_state_dict(checkpoint['model_state_dict'])
    model.threshold = checkpoint['threshold']
    
    print(f"Loaded model with optimal threshold: {model.threshold:.4f}")
    
    # Testing
    print("Testing Model .......")
    test_df = generate_singleLabel_data('../uli_dataset/testing/', 'l1')
    # test_df = singleLang_singleLabel('./uli_dataset/testing/','en','l1')
    test_ds = AbuseDataset(test_df['text'].astype(str).tolist(), test_df['label'].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=16)
    testing(model, test_loader, device)