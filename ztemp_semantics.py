#%%
import json, re, os
import pandas as pd

from pandas import DataFrame
from nlp_llm import queryGPT

#%%
df = pd.read_csv(R"temp_output.csv")

df_ap = df.dropna(subset=["Assess_Plan"])
df_ap

df_ref = df.dropna(subset=["Abstract"])
df_ref
#%%
df1 = df.dropna(subset=["Assess_Plan", "Abstract"])
file_path = R"temp_output_filt.csv"
df1.to_csv(file_path)
df1
#%%
prop_list = [
    "What is the patient's diagnosis?",
    # "Is the patient's diagnosis central stenosis, foraminal, or lateral/subarticular?",
    "What are the patient's radiological findings?",
    
    "Are the patient's radiological findings consistent with their diagnosis?",
    
    "What treatments has the patient tried already?",
    "Have the treatments tried by the patients helped?",
    "Have the patient's symptoms been progressing?",
    "Are the patient's symptoms debilitating?",
    
]
prop_list = [
    "Is the patient's pathology central canal stenosis, foraminal stenosis, lateral stenosis, or another type of pathology?",
    "Is the patient's pathology central canal stenosis, foraminal stenosis, lateral stenosis, or another type of pathology?",
    "Has the patient tried physical or physiotherapy?"
    "Has the patient tried pharmacotherapy?"
    "Has the patient tried injections?"
    "Has this patient tried any interventions?"
    
]
criteria = "\n".join([F"- {i}" for i in prop_list])

header = '| ' + ' | '.join([
                            'Question (list verbatim from system message)',
                            'Response',
                            'Detailed reasoning based on given clinical information'
                        ]) + ' |'

system_query = (F"You are a thorough clinician. You will be given a clinical information "
                F"regarding a patient with back pain. Only respond using a Markdown table "
                F"without any additional sentences outside of the Markdown table.\n\n"
                
                F"You are interested in the following questions:\n"
                F"{criteria}\n\n"
                
                F'Report your findings to the questions listed above in a Markdown table '
                F'with the following header:\n'
                F'{header}'
)
print(system_query)
#%%
df_out = DataFrame()

for ind, row in df1.iterrows():
    print(ind)
    context_input = (
        F"Referral letter:\n"
        F"{row['Abstract']}\n\n"
        F"Final assessment:\n"
        F"{row['Assess_Plan']}"
    )
    raw_response = queryGPT(context_input, system_query)
    main_response = raw_response["choices"][0]["message"]["content"]
    

    col_raw = "Raw" # Raw GPT output JSON (unformatted)
    col_response = "Response"
    col_stmts = "Statements" # Organized data elements from GPT output in a JSON
    
    try:
        stmts: list[str] = [item.strip() for item in main_response.split("\n")] # Split by newline, only include lines that have word characters
        ledger = list(filter(lambda item: re.search(r"\w", item) == None, stmts))[0]
        stmts_real = stmts[stmts.index(ledger) + 1 : ] # Get statements after ledger
        
        article_stmts_str = [item for item in stmts_real if re.search(r"\w", item) != None] # Split by newline, only include lines that have word characters
        stmts_items = [[i.strip() for i in filter(None, stm.split("|"))] for stm in stmts_real] # Filter with none to get rid of empty strings, should only return 3 items corresponding to the 3 columns of output

    except Exception as error: # Default to none so that at least item gets appended
        print("Error in parsing output: ", error)
        stmts_items = [None]
        stmts_vec1 = [None]
        stmts_vec2 = [None]

    raw_json = json.dumps(raw_response) # Shape of list[list[str]] for each article, inner lists are statements, each str is an item
    stmts_json = json.dumps(stmts_items)
    
    new_row = DataFrame({
                            col_raw: [raw_json],
                            col_response: [main_response],
                            col_stmts: [stmts_json],
                        }
                        ) # Create new row for appending 



    new_row.index = pd.RangeIndex(start=ind, stop=ind+1, step=1) # Reassign index of new row by using current index 
    df_out = pd.concat([df_out, new_row])

    print(context_input)
    
    # if ind > 78: # For testing
    #     break
df_merged = pd.concat([df1, df_out], axis = 1) # Concat on columns instead of rows
# df_merged = df_merged.dropna(subset=["Statements"])

#%%


errors = []

for ind, row in df_merged.iterrows():
    stmts_str = row["Statements"]
    if isinstance(stmts_str, str):
        try:
            stmts = json.loads(stmts_str)     
            cols = [l[0] for l in stmts] # Reload cols for every instance  
            for i, col in enumerate(cols):
                df_merged.at[ind, col] = stmts[i][1] # Take second item of each list 
        except:
            errors.append([ind, row])
new_filepath = F"{os.path.splitext(file_path)[0]}_annot.csv"
df_merged.to_csv(new_filepath)

#%%
