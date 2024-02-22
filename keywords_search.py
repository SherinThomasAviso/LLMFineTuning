import pandas as pd
import os
import json
import pandas as pd
import numpy as np
import multiprocessing
from datasets import load_dataset
from nltk.tokenize import RegexpTokenizer
import tqdm
kw_file = open('keywords.txt', 'r')
out = []
for line in kw_file.readlines():
    out= out+line.replace('\n','').lower().split(',')

kw_list = list(set(out))


def process_chunk(df_chunk):
        df_chunk['keywords'] = ''
        for index, row in df_chunk.iterrows():
            
            tokenizer = RegexpTokenizer(r'\w+')
            test_set = set(tokenizer.tokenize(row['response'].lower()))
            
            if (set(out) & test_set):
                  df_chunk.at[index,'keywords'] = list(set(kw_list) & test_set)
        
        return df_chunk.loc[df_chunk['keywords']!='']


if __name__ == '__main__':
            #use all available cores , otherwise specify the number you want as an argument,
            #for example if you have 12 cores,  leave 1 or 2 for other things
            pool = multiprocessing.Pool(processes=10) 
            chunk_size = 50
            dataset = load_dataset('databricks/databricks-dolly-15k')
            df=pd.DataFrame(dataset['train'])
            
            results = list(tqdm.tqdm(pool.imap_unordered(process_chunk, [df[c:c+chunk_size] for c in range(0,len(df),chunk_size)]), total=len(df)/chunk_size))
            #results = pool.map(process_chunk, [df[c:c+chunk_size] for c in range(0,len(df),chunk_size)])
            
            pool.close()
            pool.join()
            
            #make new df by concatenating
            concatdf = pd.concat(results, axis=0, ignore_index=True)
            print('Total records:',len(concatdf))
            concatdf.to_csv('kw_dataset_dolly.csv',index=False)
