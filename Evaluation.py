
from requests.auth import HTTPBasicAuth    
import json
import requests
import os
import pandas as pd
from ast import literal_eval
FINETUNED_LLAMA_URL = 'https://sales-finetuned-llama-2-7b-hf-llm-finetuning-8000.aviso.Truefoundry.cloud/v1/completions'
LLAMA_13B_URL = 'https://nous-hermes-llama2-13b-llm-deployment-8000.aviso.Truefoundry.cloud/v1/completions'
LLAMA_7B_URL = 'https://llama-2-7b-hf-llm-finetuning-8000.aviso.truefoundry.cloud/v1/completions'
GPT_URL = 'https://gpt-qna-prod-llm-deployment-prod-8000.ci.aviso.com/gpt_qna'

HEADERS = {'accept': 'application/json', 'Content-Type': 'application/json'}

df = pd.read_excel('clean_code/validation_data/bert_dataset_gpt.xlsx')

def create_request(model,prompt,URL):
    prompt = 'Answer the below question :\n' + prompt 
    max_tokens = 4096 - len(prompt)
    body = {
    "model": model, #"llama-2-7b-hf",#"nous-hermes-llama2-13b",
    "prompt": [prompt],
    "max_tokens": max_tokens,
    "temperature": 0.1,
    "top_p": 1,
    "n": 1,
    "stream": False,
    "logprobs": 0,
    "echo": False,
    "stop": [
        "string"
    ],
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "best_of": 1,
    
    "user": "string",
    "top_k": -1,
    "ignore_eos": False,
    "use_beam_search": False,
    "stop_token_ids": [
        0
    ],
    "skip_special_tokens": True,
    "spaces_between_special_tokens": True,
    "repetition_penalty": 1,
    "min_p": 0
    }

    response = requests.post(URL, headers=HEADERS, json=body)
    if response.status_code == 200:
        
        return (response.json()['choices'][0]['text'])
    else:
        print(response.json())
        return 'ERROR'
def create_request_gpt(question):
    prompt = 'Answer the below question :\n' + question 
    body = {"prompt": prompt}

    response = requests.post(GPT_URL, headers=HEADERS, json=body)
    if response.status_code == 200:
        return (response.json())
    else:
        print(response)
        return 'ERROR'

def get_gpt_score(question, answer1, answer2,answer3,answer4):

    prompt = """Given a sales related question below: \n
    Question : {}

    Please evaluate the following four answers with respect to the question, providing a score from 1 to 5 for each answer. 
    Note that 1 denotes a poor answer while 5 signifies an excellent one.
    Your response should simply be a single integer between 1 and 5 for each answer. 

    Here's the answers:
    Answer 1 : {}

    Asnwer 2 : {}

    Asnwer 3 : {}

    Asnwer 4 : {}
    
    Output Format:
    THE OUTPUT SHOULD BE IN THE FOLLOWING FORMAT : {{"Answer 1":"Score for Answer 1","Answer 2":"Score for Answer 2","Answer 3":"Score for Answer 3","Answer 4":"Score for Answer 4"}}

    """.format(question,answer1,answer2,answer3,answer4)
    #print(prompt)
    body = {"prompt": prompt}

    response = requests.post(GPT_URL, headers=HEADERS, json=body)
    if response.status_code == 200:
        return (response.json())
    else:
        print(response)
        return 'ERROR'


result_df = pd.DataFrame(columns = ['Question','Answer-FineTuned','Answer-Baseline-Llama 7b','Answer-Baseline-Llama 13b','Answer-Baseline - GPT','GPT-Score FineTuned',
                                    'GPT-Score Baseline - Llama 7b','GPT-Score Baseline - Llama 13b','GPT-Score Baseline - GPT'])

for i, row in df.iterrows():
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i)
    if str(row['input'])!='nan':
        question = str(row['instruction'])+'\n'+ str(row['input'])
    else:
        question = str(row['instruction'])
    print(question)
    finetuned_result = create_request('llama-2-7b-hf',question,FINETUNED_LLAMA_URL)
    print('-----------------Finetune Result :----------------- \n ',finetuned_result)
    baseline_result_llama7b = create_request('llama-2-7b-hf',question,LLAMA_7B_URL)
    print('-----------------Baseline Result Llama 7b:----------------- \n',baseline_result_llama7b)
    baseline_result_llama13b = create_request('nous-hermes-llama2-13b',question,LLAMA_13B_URL)
    print('-----------------Baseline Result Llama 13b:----------------- \n',baseline_result_llama13b)
    baseline_result_gpt = create_request_gpt(question)
    print('-----------------Baseline Result GPT :----------------- \n',baseline_result_gpt)
     
    
    
    
    try:
        gpt_score = get_gpt_score(question,finetuned_result,baseline_result_llama7b,baseline_result_llama13b,baseline_result_gpt)
        gpt_score = literal_eval(gpt_score)
        print(gpt_score)
        print('**********************************************************')
        result_df.loc[len(result_df)] = [question,finetuned_result,baseline_result_llama7b,baseline_result_llama13b,baseline_result_gpt,
                                         gpt_score['Answer 1'],gpt_score['Answer 2'],gpt_score['Answer 3'],gpt_score['Answer 4']]
        if i==200:
            break
    except Exception as e:
        print(e)
        result_df.loc[len(result_df)] = [question,"","","","","","","",""]
    result_df.to_excel('clean_code/validation_data/validation_result_gpt_4models.xlsx')