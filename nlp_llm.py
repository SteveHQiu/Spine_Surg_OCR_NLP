#%% Imports
import re, time, json, os, io, traceback

# OpenAI
import openai
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv, dotenv_values

#%%
load_dotenv()
ENV = dotenv_values(".env") # Use dotenv_values since load_dotenv seems to have some issues parsing values containing "="

if "FB_TYPE" in ENV and "FB_PRIVATE_KEY" in ENV: # For use locally 
    OPENAI_KEY = ENV['OPENAI_API_KEY']
    AZURE_KEY = ENV['AZURE_API_KEY']
else:
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    AZURE_KEY = os.getenv('AZURE_API_KEY')
    

AZURE = 1
if AZURE:
    DEPLOY_NAME = "main" # Azure Resource deployment name
    openai.api_type = "azure"
    openai.api_base = "https://openai-main-09-23.openai.azure.com/" # Azure OpenAI user resource endpoint
    openai.api_key = AZURE_KEY
    openai.api_version = "2023-05-15"
else:
    openai.api_key = OPENAI_KEY


#%%

def queryGPT(query, system_query = None, model = "gpt-3.5-turbo"):
    # model = "gpt-4"
    # model = "gpt-3.5-turbo-16k"

    output_raw = None
    messages = [{"role": "user", "content": query},]
    if system_query:
        messages.insert(0, {"role": "system", "content": F"{system_query}"})

    print(messages)
    try:
        if AZURE:
            output_raw = openai.ChatCompletion.create(
                messages=messages,
                temperature=0,
                n=1,
                engine=DEPLOY_NAME, # For Azure only
                )
        else: # Defaul to standard OpenAI API 
            output_raw = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                n=1,
                )
            
        if output_raw != None:
            print(F"Received response")

    except (openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError) as e:
        print(f"OpenAI API error: {e}\nRetrying...")
        time.sleep(1.5)
        output_raw = queryGPT(query=query, system_query=system_query, model=model)
        
    return output_raw
#%%
