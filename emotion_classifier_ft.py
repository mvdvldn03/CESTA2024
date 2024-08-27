import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import glob
import re

class TextDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

tokenizer = AutoTokenizer.from_pretrained("bergum/xtremedistil-l6-h384-go-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bergum/xtremedistil-l6-h384-go-emotion")
device = torch.device('cpu')
model.to(device)

directory_path = '/Users/maxv/Downloads/paragraph_data/xlsx/*.xlsx'
file_paths = glob.glob(directory_path)

dfs = []
for path in file_paths:
    print(path)
    df = pd.read_excel(path, engine='openpyxl')
    file_name = re.search(r'[^\/]*\/([^\/]+)\.xlsx', path).group(1)
    print([x for x in file_name.split("_") if x])
    df[['author', 'cqp', "X1", "date", "X2", 'para']] = [x for x in file_name.split("_") if x]
    dfs.append(df.drop(columns=['cqp', 'para']))

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['paragraph'])
print(df.shape)
sentences = df['paragraph'].tolist()
print(df)
df[['author', "date", "X1", "X2", 'paragraph', 'plnames', 'geonouns']].to_csv('final_paragraphs.csv')

'''
max_length = 128
dataset = TextDataset(sentences, tokenizer, max_length)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

i = 0
model.eval()
probs = np.zeros((0, 28))
preds = np.zeros((0, 28))
with torch.no_grad():
    try:
        for batch in loader:
            if i % 100 == 0:
                print(i)

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            batch_probs = torch.sigmoid(logits)
            max_idx = torch.argmax(batch_probs, dim=1)
            batch_preds = torch.zeros_like(batch_probs)
            batch_preds[torch.arange(batch_probs.size(0)), max_idx] = 1

            #print([round(x, 3) for x in batch_probs.tolist()[0]])
            probs = np.vstack((probs, batch_probs))
            preds = np.vstack((preds, batch_preds))
            i += 1
    except TypeError as e:
        print(e)
        print("Data types:", input_ids.dtype, attention_mask.dtype)
        # Check for correct device placement if using CUDA
        print("Device:", input_ids.device, attention_mask.device)
        # Check shape alignment
        print("Shapes:", input_ids.shape, attention_mask.shape)
        print(df.iloc[i])

columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiousity', 'desire',
           'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
           'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'suprise', 'neutral']
result = pd.DataFrame(probs, columns=columns)
result.insert(0, 'sentence', sentences)
result.to_csv('full_28_probs.csv', index=False)
'''