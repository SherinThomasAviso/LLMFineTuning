import pandas as pd
from nltk.tokenize import RegexpTokenizer
dataset = ['alpaca','wizardlm','dolly','ign']
input_path = 'clean_code/Leftover_Data/leftover_raw/bert_dataset_'
output_path = 'clean_code/Leftover_Data/leftover_processed/bert_dataset_'
dataset = ['gpt']
input_path = 'clean_code/validation_data/bert_dataset_'
output_path = 'clean_code/validation_data/bert_dataset_'
for d in dataset:
        print('-------',d,'-------')
        filename = d+'_kw_leftover.csv'
        df = pd.read_csv(input_path+filename)

        df = df.sort_values(by=['prob'],ascending=False)
        print('Before:',len(df))

        kw_list = ['sales','business','deal','customer','marketing','salesforce','salesperson','customers','revenue','purchase','product','service','services']
        code_kw = ['excel','python','sql','ruby','r','database']

        df['sales_keywords'] = ''
        df['code_keywords'] = ''
        for index, row in df.iterrows():
                
                tokenizer = RegexpTokenizer(r'\w+')
                if d=='dolly':
                        test_set = set(tokenizer.tokenize(row['response'].lower()))
                else:
                        test_set = set(tokenizer.tokenize(row['output'].lower())) 
                
                if (set(kw_list) & test_set):
                        df.at[index,'sales_keywords'] = list(set(kw_list) & test_set)

                if (set(code_kw) & test_set):
                        df.at[index,'code_keywords'] = list(set(code_kw) & test_set)
                        df.at[index,'code_keywords'] = list(set(code_kw) & test_set)
                        df.at[index,'code_keywords'] = list(set(code_kw) & test_set)

        
        df = df.loc[(df['code_keywords']=='')&(len(df['sales_keywords'])!='')]
        df = df.loc[df['sales_keywords'].map(len)>1]
        print('After:',len(df))
        #df.to_csv(filename,index=False)
        df.to_excel(output_path+filename.replace('.csv','.xlsx'),index=False)
        