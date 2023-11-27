#%%
import os, re
from dataclasses import dataclass
#%%
# Internals 
# from ocr import getPdfPages, extractText, extractTextBlock
from nlp_spacy import anonText
from pdf import processPDF
#%%
from pdf import processPDF
import re

indexes = []
texts = []
errors = []

for i in range(3, 7):
    dir_path = RF"C:/Users/steve/OneDrive/Spine_Surg_OCR_NLP/data/Group {i}/"
    print(dir_path)
    for file in os.listdir(dir_path):
        if file.endswith("pdf") and "referral" in file:
            print(file)
            pt_no = re.search(R"(\d+)_referral", file)
            try:
                texts.append(processPDF(F"{dir_path}{file}"))
                indexes.append(pt_no.group(1))
            except:
                print("ERROR")
                errors.append(file)

indexes = [int(i) for i in indexes]
#%%
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
temp_path = "temp_texts.dat"
#%%
save_object([indexes, texts], temp_path)
#%%
indexes, texts = load_object(temp_path)
#%%
from nlp_spacy import anonText
texts_anon = [anonText(t) for t in texts]

#%%
import pandas as pd

df = pd.DataFrame({"Study ID": indexes, "Abstract": texts_anon})
df.to_csv("temp_df.csv")
df_base = pd.read_excel("data_combined.xlsx")
combined = pd.merge(df_base, df, on="Study ID", how="left")
combined.to_csv("temp_output.csv")
#%%
import pandas as pd
combined = pd.read_csv("temp_output.csv")
combined = combined.dropna(subset=["Abstract"])

#%%
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def extract_unique_words_and_ngrams(documents, n=2):
    unique_words = set()
    unique_ngrams = set()

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for doc in documents:
        # Tokenize the document
        tokens = word_tokenize(doc.lower())
        
        tokens = [word for word in tokens if word.isalnum()]
        
        # Extract n-grams before removing stop words
        for i in range(2, n + 1):
            n_grams = ngrams(tokens, i)
            unique_ngrams.update(n_grams)
            
        # Remove stopwords
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Extract unique words
        unique_words.update(tokens)
        
    
    return list(unique_words), list(unique_ngrams)

# Example usage

documents = combined["Abstract"]

unique_words, unique_ngrams = extract_unique_words_and_ngrams(documents, n=3)
#%%
concat_ngrams = [" ".join(g) for g in unique_ngrams]
all_terms = unique_words + concat_ngrams
assert all([isinstance(a, str) for a in all_terms])
#%%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


df = combined

# Define a list of words or phrases you want to analyze
keywords = all_terms

# Create a new column for each keyword to indicate presence or absence
for keyword in keywords:
    try:
        df[keyword] = df['Abstract'].str.contains(keyword, case=False)
    except:
        print(F"Error with {keyword}")
#%%
# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Keyword', 'Chi-Square', 'P-Value', 'Association', "OddsRatio"])

for keyword in keywords:
    try:
        contingency_table = pd.crosstab(df[keyword], df['Surgery_Final'])
        
        # https://www.ncbi.nlm.nih.gov/books/NBK431098/figure/article-37432.image.f1/
        a = contingency_table.loc[True].loc[1] # Yes keyword, yes surgery
        b = contingency_table.loc[True].loc[0] # Yes keyword, no surgery 
        c = contingency_table.loc[False].loc[1] # No keyword, yes surgery
        d = contingency_table.loc[False].loc[0] # No keyword, no surgery 
        odds_ratio = (a*d)/(b*c)
        
        chi2, p, _, _ = chi2_contingency(contingency_table)
        association = "Significant" if p < 0.05 else "Not Significant"
        results_df = pd.concat([results_df, pd.DataFrame([[keyword, chi2, p, association, odds_ratio]], columns=results_df.columns)])
    except:
        print(F"Error with {keyword}")
# Reset the index of the results DataFrame
results_df.reset_index(drop=True, inplace=True)
#%%

df_prelim = results_df.loc[results_df["P-Value"] < 0.05]
df_prelim.sort_values(by=["P-Value"], inplace=True)
df_prelim.reset_index(drop=True, inplace=True)
df_prelim.to_csv("prelim_ngrams2.csv")
#%%
results_df.sort_values(by=["P-Value"])
results_df.to_csv("prelim_word_ngrams2.csv")
# %%
