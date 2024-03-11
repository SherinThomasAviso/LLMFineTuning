import trafilatura
import requests
import json
import ast
import pandas as pd

GPT_MODEL_URL ="https://gpt-qna-prod-llm-deployment-prod-8000.ci.aviso.com/gpt_qna"
HEADERS = {'accept': 'application/json', 'Content-Type': 'application/json'}

def scrape_text(URL):
    downloaded = trafilatura.fetch_url(URL)
    text =  trafilatura.bare_extraction(downloaded)['text']
    return text

def generate_qa(text):
    prompt = '''You are an AI assistant tasked with generating question and answer pairs for the given context using the given format. Only answer in the format with no other text. You should create question/answer pairs. Return the question/answer pairs as a Python List. Each dict in the list should have the full context provided, a relevant question to the context and answer to the question with minimum 3 sentences.Give complete answer for each question. Create a minimum 0f 10 question/answer pairs and maximum of 25 question/answer pairs.

    Format:
    [{{"Input": str, "Output": str}}]
    Follow the output format obidiently.
    Return the question/answer pairs as a Python List only.DO NOT ADD ANY SERIAL NUMBER FOR THE PAIRS.

    Context:
    {}

    '''.format(text)
    data = {"prompt":prompt}
    response = requests.post(GPT_MODEL_URL, headers=HEADERS, json= data)
    #print(response)
    if response.status_code==200:
        qa = response.json()
        return qa
    else:
        print('Error in response',response.status_code)


if __name__=="__main__":
    
    url_list = ['https://avenuetalentpartners.com/2021/01/27/what-is-enterprise-sales-complex-sales/',
    "https://www.daniel-one.com/design-sales-process-funnel-b2b",
    "https://www.pipedrive.com/en/blog/sales-methodology#how-to-choose-the-right-methodology-for-your-business",
    "https://www.getcompass.ai/glossary/sales-analytics",
    "https://www.zendesk.com/blog/sales-process/","https://www.iovox.com/blog/conversation-intelligence#:~:text=Conversation%20intelligence%20captures%20and%20analyzes,their%20preferences%20and%20search%20patterns.","https://encharge.io/what-is-revenue-intelligence/"]
    for URL in url_list:
        print(URL)
        out_df = pd.read_csv('qa_dataset.csv')
        text = scrape_text(URL)
        qa = generate_qa(text)
    
        if URL not in out_df['URL']:
            for pair in ast.literal_eval(qa):
                out_df.loc[len(out_df)]= [pair['Input'],pair['Output'],URL]
        out_df.to_csv('qa_dataset.csv',index=False)