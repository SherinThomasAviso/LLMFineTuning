import tqdm
import torch
import numpy as np
import pandas as pd
from torch import nn
import multiprocessing
import torch.nn.functional as F
from multiprocessing import set_start_method
from datasets import load_dataset
from transformers import BertTokenizer, BertModel


# Set up parameters globally
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 512
batch_size = 16
num_epochs = 2
learning_rate = 2e-5
fine_tuned_model_path = 'bert_classifier.pth'

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits


def predict_class(text, model, tokenizer, device, max_length=512):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = F.softmax(outputs, dim=-1)
            prob, preds = torch.max(outputs, dim=1)
                        
    if preds.item() == 1:
          if prob.item()>=0.5:
                return 'sales',prob.item()
    return 'other',prob.item()

def process_chunk(df_chunk):
        df_chunk['pred_lbl'] = ''
        df_chunk['prob'] = ''
        
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTClassifier(bert_model_name, num_classes).to(device)
        model.load_state_dict(torch.load(fine_tuned_model_path))

        for i,row in df_chunk.iterrows(): 
            lbl,prob = predict_class(row['response'], model, tokenizer, device)
            df_chunk.at[i,'pred_lbl'] = lbl
            df_chunk.at[i,'prob'] = prob
        return df_chunk.loc[df_chunk['pred_lbl']!='other']



if __name__ == '__main__':
            hf_dataset_name = 'databricks/databricks-dolly-15k'
            set_start_method('spawn')
            pool = multiprocessing.Pool(processes=10) #use all available cores , otherwise specify the number as an argument
            chunk_size = 5 #specify number of records to be processed in one thread at a time
            dataset = load_dataset(hf_dataset_name)
            df=pd.DataFrame(dataset['train'])[:50]
            #df = pd.read_csv('Dataset1_KeywordSearch.csv')
            

            
            results = list(tqdm.tqdm(pool.imap_unordered(process_chunk, [df[c:c+chunk_size] for c in range(0,len(df),chunk_size)]), total=len(df)/chunk_size))
            #results = pool.map(process_chunk, [df[c:c+chunk_size] for c in range(0,len(df),chunk_size)])
            pool.close()
            pool.join()
            
            #make new df by concatenating
            
            concatdf = pd.concat(results, axis=0, ignore_index=True)
            #print(concatdf)
            concatdf.to_csv(hf_dataset_name.split('/')[1]+'.csv',index=False)
