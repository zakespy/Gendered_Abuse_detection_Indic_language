import re
import os
import html
import emoji
import numpy as np
import pandas as pd 


annotator_dict = {
    'en' : ['en_a1', 'en_a2', 'en_a3', 'en_a4', 'en_a5', 'en_a6'],
    'hi' : ['hi_a1', 'hi_a2', 'hi_a3', 'hi_a4', 'hi_a5'],
    'ta' : ['ta_a1', 'ta_a2', 'ta_a3', 'ta_a4', 'ta_a5', 'ta_a6']
}


def get_max_count_label(row):
        counts = {0: 0, 1: 0}
        for val in row:
            if pd.notna(val):  # Ignore NaN values
                if isinstance(val, (int, float)):  # Ensure numeric type
                    if val == 0.0 or val == 0:
                        counts[0] += 1
                    elif val == 1.0 or val == 1:
                        counts[1] += 1

        if counts[1] > counts[0]:  # More abusive votes
            return 1
        elif counts[0] > counts[1]:  # More non-abusive votes
            return 0
        else:  
            return 1 

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'@\w+', 'user', text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\u0B80-\u0BFF\s.,!?]", "", text)  
    text = re.sub(r"\s+", " ", text).strip() 
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'[!\"“”‘’\'”]+', '', text)
    text = html.unescape(text)
    text = emoji.demojize(text,language='en')
    text = re.sub(r':([a-zA-Z0-9_]+):', lambda m: m.group(1).replace('_', ' '), text) # replacing colon in demojized emojie with space
    return text


# Function to try different CSV parsing methods
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


def singleLang_singleLabel(source,lang,label):
    newData = pd.DataFrame()
    files = os.listdir(source)
    
    for file in files:
        if file.split('.')[0].split('_')[-1] == label and file.split('.')[0].split('_')[1] == lang:
            data = pd.read_csv(source + file)
            data['lang'] = lang
            # print(lang)
            annotator_cols = annotator_dict[lang]
            # print(data.head())
            # print(data.columns)
            data[annotator_cols] = data[annotator_cols].replace("NAN", np.nan).apply(pd.to_numeric, errors="coerce")
            data['label'] = data[annotator_cols].apply(get_max_count_label, axis=1)
            data = data[['text','lang', 'label']].dropna()
            data['text'] = data['text'].str.replace('<handle replaced>', '', regex=False)
            data["text"] = data["text"].apply(clean_text)
            newData = pd.concat([newData, data], axis=0)
    newData.reset_index(drop=True, inplace=True)
    return newData
    


def generate_singleLabel_data(source, label):
    
    newData = pd.DataFrame()
    
    files = os.listdir(source)
    
    for file in files:
        if file.split('.')[0].split('_')[-1] == label:
            data = pd.read_csv(source + file)
            lang = file.split('_')[1]  # Extract language (en, hi, ta)
            data['lang'] = lang
            # print(lang)
            annotator_cols = annotator_dict[lang]
            # print(data.head())
            # print(data.columns)
            data[annotator_cols] = data[annotator_cols].replace("NAN", np.nan).apply(pd.to_numeric, errors="coerce")
            data['label'] = data[annotator_cols].apply(get_max_count_label, axis=1)
            data = data[['text','lang', 'label']].dropna()
            data['text'] = data['text'].str.replace('<handle replaced>', '', regex=False)
            data["text"] = data["text"].apply(clean_text)
            newData = pd.concat([newData, data], axis=0)
    newData.reset_index(drop=True, inplace=True)
    return newData


def generate_multilabel_data(source, labelList,destingationPath = None):
    merged_df = pd.DataFrame()
    dfs = []
    
    for label in labelList:

        newData = pd.DataFrame()
        files = os.listdir(source)

        for file in files:
            if file.split('.')[0].split('_')[-1] == label:
                data  = load_csv_with_fallbacks(source + file)
                lang = file.split('_')[1]  # Extract language (en, hi, ta)
                data['lang'] = lang
                # print(lang)
                annotator_cols = annotator_dict[lang]
                # print(data.head())
                # print(data.columns)
                data[annotator_cols] = data[annotator_cols].replace("NAN", np.nan).apply(pd.to_numeric, errors="coerce")
                data['label'] = data[annotator_cols].apply(get_max_count_label, axis=1)
                data = data[['text','lang','key','label']].dropna()
                data['text'] = data['text'].str.replace('<handle replaced>', '', regex=False)
                data["text"] = data["text"].apply(clean_text)
                newData = pd.concat([newData, data], axis=0)
        newData.reset_index(drop=True, inplace=True)
        
        dfs.append(newData)
    
    for index,df in enumerate(dfs):
        df.rename(columns={'key': f'key_{index+1}', 'label': f'label_{index+1}'}, inplace=True)
    
    # Merge all dataframes on 'text' and 'lang'
    for df in dfs:
        merged_df = pd.merge(merged_df, df, on=['text', 'lang'], how='outer') if not merged_df.empty else df
    
    for i in range(len(dfs)):
        merged_df[f'key_{i+1}'].fillna('', inplace=True)
        merged_df[f'label_{i+1}'].fillna(0, inplace=True)
    # merged_df.to_csv(os.path.join(destingationPath, 'merged.csv'), index=False)
    return merged_df
        

def dividing_en_data(trainPath,testPath,filePath):
    # files = os.listdir(trainPath)
    # df = pd.DataFrame()
    # for file in files:
    #     print(file)
    #     if(file == fileName):
    #         df = pd.read_csv(trainPath + file)
    #         break
    
    df = pd.read_csv(filePath)
    
    def classification(label):
        if label == 0 or label == 1:
            retlabel = 0
        else :
            retlabel = 1
        return retlabel
    
    print(df.head())
    df['class'] = df['class'].apply(classification)
    df.rename(columns={'class': 'label','tweet':'text'}, inplace=True)
    df = df[['text','label']].dropna()
    # df['text'] = df['text'].str.replace('<handle replaced>', '', regex=False)
    # df["text"] = df["text"].apply(clean_text)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataframe
    split_index = 18000  # Split into two equal parts
    df1 = df.iloc[:split_index]  # First half
    df2 = df.iloc[split_index:]  # Second half
    
    # Save the dataframes to the testPath folder
    df1.to_csv(os.path.join(trainPath, 'english_train.csv'), index=False)
    df2.to_csv(os.path.join(testPath, 'english_test.csv'), index=False)
    
    return df
    
def get_pretrained_data(source,destinationPath= None,abusiveLabel = 1):
    files = os.listdir(source)
    merged_df = pd.DataFrame()
    
    def classification(label):
        if label == 0:
            retlabel = abusiveLabel
        else:
            retlabel = 1-abusiveLabel
        return retlabel
    
    for file in files:
        # if file.split('.')[0].split('_')[1] == keyword:
            df = pd.read_csv(source + file)
            df['text'] = df['text'].str.replace('<handle replaced>', '', regex=False)
            df["text"] = df["text"].apply(clean_text)
            df['label'] = df['label'].apply(classification)
            merged_df = pd.merge(merged_df, df, on=['text', 'label'], how='outer') if not merged_df.empty else df
    
    # merged_df.to_csv(os.path.join(destinationPath, 'pretrained_test.csv'),index=False)
    # print(merged_df['label'])
    return merged_df
    
def calculate_class_count(path):
    dataset = pd.read_csv(path)
    
    class_counts = dataset['label'].value_counts().to_dict()
    return class_counts
    
if __name__ == '__main__':
    # print(generate_singleLabel_data('./uli_dataset/training/','l1'))
    # print(generate_multilabel_data('./uli_dataset/training/',['l1','l3'],'./uli_dataset/'))
    # print(dividing_en_data('./pretrainingData/training/','./pretrainingData/testing/','./pretrainingData/labeled_data.csv'))
    print(get_pretrained_data('./pretrainingData/testing/','./pretrainingData/',abusiveLabel=1))
    # print(calculate_class_count('./pretrainingData/pretrained_test.csv'))
    # print(get_pretrained_data('./pretrainingData/training/'))
    