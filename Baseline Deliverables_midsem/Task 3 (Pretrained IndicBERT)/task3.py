import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split


def get_max_count_label(row):
    # row :- dataframe row 
    # Calculating text label from all  6 annotators
    
        counts = {0: 0, 1: 0}
        for val in row:
            if pd.notna(val): 
                if isinstance(val, (int, float)): 
                    if val == 0.0 or val == 0:
                        counts[0] += 1
                    elif val == 1.0 or val == 1:
                        counts[1] += 1

        if counts[1] > counts[0]:  
            return 1
        elif counts[0] > counts[1]: 
            return 0
        else:  
            return 1 

def clean_text(text):
    # Text :- string test 
    #  cleaning text by lowering the text, removing placeholder and urls 
    text = text.lower()  
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\u0B80-\u0BFF\s.,!?]", "", text)  
    text = re.sub(r"\s+", " ", text).strip() 
    text = re.sub(r'<.*?>', ' ', text)
    return text


annotator_dict = {
    'en' : ['en_a1', 'en_a2', 'en_a3', 'en_a4', 'en_a5', 'en_a6'],
    'hi' : ['hi_a1', 'hi_a2', 'hi_a3', 'hi_a4', 'hi_a5'],
    'ta' : ['ta_a1', 'ta_a2', 'ta_a3', 'ta_a4', 'ta_a5', 'ta_a6']
}

def load_csv_with_fallbacks(file_path):
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
            # df['language'] = language
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
                            # df['language'] = language
                            return df
                except Exception as manual_e:
                    print(f"  Manual approach failed: {str(manual_e)}")
    
    print(f"All attempts to read {file_path} failed.")
    return None




def create_df(df,lang):
    # cleaning the dataframe text and adding new column label 
    # annotator_cols = ['en_a1', 'en_a2', 'en_a3', 'en_a4', 'en_a5', 'en_a6']
    annotator_cols = annotator_dict[lang]
    df[annotator_cols] = df[annotator_cols].replace("NAN", np.nan).apply(pd.to_numeric, errors="coerce")
    df['label'] = df[annotator_cols].apply(get_max_count_label, axis=1)
    # df['label'] = df.apply(lambda row: label_mapping[(row['key'], row['label'])], axis=1)
    df = df[['text','key', 'label']].dropna()
    df['text'] = df['text'].str.replace('<handle replaced>', 'user', regex=False)
    df["text"] = df["text"].apply(clean_text)
    return df

def merge_csv(file1, file2, output_file):
    #  merging label 1 and label 3 dataset to create a single csv file which contain columns text, Key_1 , Key_2 , label_2 ,label_2
    
    # df1 = pd.read_csv(file1)
    df1 = load_csv_with_fallbacks(file1)
    df2 = load_csv_with_fallbacks(file2)
    # df2 = pd.read_csv(file2)
    
    df1 = create_df(df1,file1.split('_')[1])
    df2 = create_df(df2,file2.split('_')[1])
    

    df1.rename(columns={'key': 'key_1', 'label': 'label_1'}, inplace=True)
    df2.rename(columns={'key': 'key_2', 'label': 'label_2'}, inplace=True)
    
    
    merged_df = pd.merge(df1, df2, on='text', how='outer')
    

    merged_df['key_1'].fillna('', inplace=True)
    merged_df['key_2'].fillna('', inplace=True)
    merged_df['label_1'].fillna(0, inplace=True)
    merged_df['label_2'].fillna(0, inplace=True)
    
    # Create final label column (1 if any label is 1, else 0)
    # merged_df['label'] = merged_df[['label_1', 'label_2']].max(axis=1)
    
    # Drop intermediate label columns
    # merged_df.drop(columns=['label_1', 'label_2'], inplace=True)
    
    # Save to output CSV
    merged_df.to_csv(output_file, index=False)
    
    print(f'Merged CSV saved to {output_file}')



# Dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])
        key_1 = 1 if self.data.iloc[idx]["key_1"] else 0
        key_2 = 1 if self.data.iloc[idx]["key_2"] else 0
        label_1 = self.data.iloc[idx]["label_1"]
        label_2 = self.data.iloc[idx]["label_2"]

        
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "key_1": torch.tensor(key_1, dtype=torch.float),
            "key_2": torch.tensor(key_2, dtype=torch.float),
            "label_1": torch.tensor(label_1, dtype=torch.float),
            "label_2": torch.tensor(label_2, dtype=torch.float),
            "text": text  
        }


# Classification Model
class IndicBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super(IndicBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.dropout(output)
        return self.fc1(output).squeeze(-1), self.fc2(output).squeeze(-1)  # Two separate outputs for key_1 and key_2

# Training Function
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=3):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0

        # Training Loop
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_1 = batch["label_1"].to(device)
            label_2 = batch["label_2"].to(device)

            output_1, output_2 = model(input_ids, attention_mask)

            loss_1 = criterion(output_1, label_1)
            loss_2 = criterion(output_2, label_2)
            loss = loss_1 + loss_2

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label_1 = batch["label_1"].to(device)
                label_2 = batch["label_2"].to(device)

                output_1, output_2 = model(input_ids, attention_mask)

                loss_1 = criterion(output_1, label_1)
                loss_2 = criterion(output_2, label_2)
                loss = loss_1 + loss_2

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Plot Training & Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o", linestyle="-", color="b", label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, marker="o", linestyle="-", color="r", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
    
    return train_losses, val_losses

# Evaluation Function
def evaluate_model(model, dataloader, device,output_csv="res_task3_merged_en.csv"):
    model.eval()
    all_preds_1, all_preds_2, all_labels_1, all_labels_2, all_texts = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_1 = batch["label_1"].to(device)
            label_2 = batch["label_2"].to(device)
            texts = batch["text"] 

            output_1, output_2 = model(input_ids, attention_mask)

            preds_1 = (torch.sigmoid(output_1) > 0.5).int().cpu().numpy()
            preds_2 = (torch.sigmoid(output_2) > 0.5).int().cpu().numpy()

            all_preds_1.extend(preds_1)
            all_preds_2.extend(preds_2)
            all_labels_1.extend(label_1.cpu().numpy())
            all_labels_2.extend(label_2.cpu().numpy())
            all_texts.extend(texts)  
    
    pred_df = pd.DataFrame({
        "text": all_texts,
        "key_1":all_labels_1,
        "predicted_label_1": all_preds_1,
        "key_2":all_labels_2,
        "predicted_label_2": all_preds_2
    })
    pred_df.to_csv(output_csv, index=False)
    f1_score_1 = f1_score(all_labels_1, all_preds_1, average="macro")
    f1_score_2 = f1_score(all_labels_2, all_preds_2, average="macro")
    macro_f1 = (f1_score_1 + f1_score_2) / 2

    return f1_score_1, f1_score_2, macro_f1

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    MODEL_NAME = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # creating single dataset for training and testing 
    merge_csv('train_en_l1.csv','train_en_l3.csv','train_task3_merged_en.csv')
    merge_csv('test_en_l1.csv','test_en_l3.csv','test_task3_merged_en.csv')
    merge_csv('train_hi_l1.csv','train_hi_l3.csv','train_task3_merged_hi.csv')
    merge_csv('test_hi_l1.csv','test_hi_l3.csv','test_task3_merged_hi.csv')
    merge_csv('train_ta_l1.csv','train_ta_l3.csv','train_task3_merged_ta.csv')
    merge_csv('test_ta_l1.csv','test_ta_l3.csv','test_task3_merged_ta.csv')

    pdf1 = pd.read_csv('train_task3_merged_en.csv')
    pdf2 = pd.read_csv('train_task3_merged_hi.csv')
    pdf3 = pd.read_csv('train_task3_merged_ta.csv')
    pdf = pd.concat([pdf1, pdf2, pdf3], ignore_index=True)
    pdf.to_csv('train_task3_merged.csv', index=False)
    
    pdf1 = pd.read_csv('test_task3_merged_en.csv')
    pdf2 = pd.read_csv('test_task3_merged_hi.csv') 
    pdf3 = pd.read_csv('test_task3_merged_ta.csv')
    pdf = pd.concat([pdf1, pdf2, pdf3], ignore_index=True)     
    pdf.to_csv('test_task3_merged.csv', index=False)
    
    # Load data
    df = pd.read_csv('train_task3_merged.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_df, tokenizer)
    val_dataset = TextDataset(val_df, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    

    # model, optimizer and loss funciton initialization 
    model = IndicBERTClassifier(MODEL_NAME).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Training model 
    num_epochs = 3
    # train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs)
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # saving model 
    # torch.save(model.state_dict(), 'task3model.pth')
    # tokenizer.save_pretrained('task3tokenizer')
    
    # loading test data
    # df = pd.read_csv('test_task3_merged.csv')
    df = load_csv_with_fallbacks('test_task3_merged.csv')
    test_dataset = TextDataset(df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    # loading model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('task3tokenizer')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndicBERTClassifier(MODEL_NAME).to(device)
    model.load_state_dict(torch.load('task3model.pth', map_location=device))
    
    # Evaluate model
    f1_1, f1_2, macro_f1 = evaluate_model(model, test_dataloader, device)
    print(f"F1-score (Key_1): {f1_1:.4f}, F1-score (Key_2): {f1_2:.4f}, Macro F1-score: {macro_f1:.4f}")

if __name__ == "__main__":
    main()



    

