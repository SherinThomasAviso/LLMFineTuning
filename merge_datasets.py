import pandas as pd
import os

name_map_dict = {'alpaca':{'output':'response','input':'context'},
                 'ign':{'output':'response','input':'instruction','instruction':'context'},
                 'wizardlm':{'output':'response'}}


def merge_dfs(data_path,df_out):
    required_cols = ['instruction', 'input', 'output','context','response']
    
    for fname in os.listdir(data_path): 
        f_path = os.path.join(data_path,fname)
        df = pd.read_excel(f_path)

        sel_cols = []
        for col in df.columns:
            if col in required_cols:
                sel_cols.append(col)
        df = df[sel_cols]
        
        #Name mapping
        
        for dataset in name_map_dict.keys():
            if dataset in fname:
            
                for col in df.columns:
                    if col in name_map_dict[dataset].keys():
                        
                        df.rename(columns = {col:name_map_dict[dataset][col]},inplace=True)
                        
            
                if dataset=='wizardlm':
                    df['context']=''
                if dataset=='ign':
                    df['context']=''
    
        df  = df[['instruction','context','response']]
        df_out.append(df)
        print(fname)
        print(len(df))
    return df_out

df_out = []
data_path = 'clean_code/Leftover_Data/leftover_processed'
df_out= merge_dfs(data_path,df_out) 
data_path = 'clean_code/Common_Data'
df_out= merge_dfs(data_path,df_out) 
df_out

df_out = pd.concat(df_out,ignore_index=True)
print(len(df_out))
print(df_out.columns)
df_out.to_excel('clean_code/Final_Train_Data.xlsx',index=False)