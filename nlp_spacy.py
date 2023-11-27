#%% Imports
# General 
import os, re, json
from collections import Counter
from typing import Union


# NLP
import spacy 
from spacy import displacy
from spacy.tokens import Span, Doc, DocBin


# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np

# Internals 
from internals import importData, LOG
#%% NLP constants
NLP = spacy.load("en_core_web_lg")

#%%


class SpacyModel:
    # Base class for other classes using SpaCy functions 
    def __init__(self, model: str = "en_core_web_sm", disable: list[str] = []):
        self.model = model
        self.NLP = spacy.load(model, disable=disable)
        self.doclist: list[Doc] = [] # List of processed text, directly usable 
        self.df_root_name: str = "" # Tracker of source path (no extension) for last import to docbin for export naming 
    
    @classmethod
    def printAvailableModels(cls):
        LOG.info(F"en_core_sci_scibert: {spacy.util.is_package('en_core_sci_scibert')}")
        LOG.info(F"en_core_sci_lg: {spacy.util.is_package('en_core_sci_lg')}")
        LOG.info(F"en_core_web_trf: {spacy.util.is_package('en_core_web_trf')}")
        LOG.info(F"en_core_web_sm: {spacy.util.is_package('en_core_web_sm')}")
        LOG.info(F"en_core_web_lg: {spacy.util.is_package('en_core_web_lg')}")
    
    def parseCorpora(self,
                      corpora_path: Union[str, bytes, os.PathLike],
                      col: str,
                      annotation_cols: list[str] = [],
                      export_userdata: bool = False,
                      **kwargs,
                      ):
        """Imports Excel or CSV file with corpora in the specified column and processes 
        it using the loaded SpaCy model. Processed results are appended to the class 
        instance in both doclist (for immediate use) and docbin (for export)
        Is the main function for batch processing texts

        Args:
            corpora_path: path to the dataframe containing the corpora
            col: Column label for column containing text to parse 
            annotation_cols: Optional list of column labels of columns to be passed as annotations to the processed text, data added to doc.user_data as dict
        """
        df = importData(corpora_path, screen_dupl=[col], screen_text=[col], **kwargs) # Screen for duplicates and presence of text for the column containing the text 
        self.df_root_name = os.path.splitext(corpora_path)[0] # Assign path without extension to tracker in case it's needed for export naming
        
        df_userdata = DataFrame()
        
        counter = 0
        for ind, row in df.iterrows():
            text: str = row[col]
            context = dict()
            for annot_col in annotation_cols: # Will not loop if list is empty 
                context[annot_col] = row[annot_col] # Add new entry for every annotation column by using data from corresponding column cell of the current row
                
            doc = self.NLP(text)
            
            # Parse user data from pipelines, assumes that all objects within user_data is JSON serializable 
            user_data_dict = {key: [json.dumps(value)] 
                         for key, value in doc.user_data.items()} # Package userdata values in list so that they can be instantiated in DataFrame
            new_entry = DataFrame(user_data_dict)
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_userdata = pd.concat([df_userdata, new_entry])
            
            doc.user_data.update(context) # Add annotation columns to .user_data attribute after user data from pipeline has been extracted
            
            self.doclist.append(doc) # Add processed doc to list for immediate use
            counter += 1
            LOG.debug(F"NLP processing text no: {counter}")
            if counter % 10 == 0: 
                LOG.info(F"NLP processing text no: {counter}") # Only print out every 10 texts
        
        if export_userdata:
            df_merged = pd.concat([df, df_userdata], axis=1)
            df_merged.to_csv(F"{self.df_root_name}_userdata.csv", index=False)
        
        
        return self.doclist
    
    def exportDocs(self, custom_name: Union[str, bytes, os.PathLike] = ""):
        """Exports current Doclist object to a file. Names export based on last import, can override this behaviour by passing a custom name

        Args:
            custom_name : Manually set file path prefix (directory and file name without extension) for the save file, extension will be added in function.
        """
        docbin = DocBin(store_user_data=True) # Need store_user_data to save Doc.user_data
        for doc in self.doclist:
            docbin.add(doc) # Serielize processed doc for exporting
            
        if custom_name: # If custom name not empty, use custom name
            docbin.to_disk(f"{custom_name}({self.model}).spacy") # Saves content using hashes based on a model's vocab, will need this vocab to import it back
            LOG.info(f"Exported DocBin to {custom_name}({self.model}).spacy")
        else: # Otherwise use prefix of last import to name output
            docbin.to_disk(f"{self.df_root_name}({self.model}).spacy") # Saves content using hashes based on a model's vocab, will need this vocab to import it back
            LOG.info(f"Exported DocBin to {self.df_root_name}({self.model}).spacy")
            
            

    def importDocs(self, path: Union[str, bytes, os.PathLike]):
        """Imports processed docs from a docbin .spacy file
        Adds imported docs to doclist

        Args:
            path (Union[str, bytes, os.PathLike]): Path to docbin object
        """
        docbin = DocBin().from_disk(path) # Don't need to set store_user_data=True for import
        doclist = list(docbin.get_docs(self.NLP.vocab)) # Retrieves content by mapping hashes back to words by using the vocab dictionary 
        self.doclist += doclist # Concat imported doclist to current doclist in case exports were done in batches
        
        
    @staticmethod # Don't need to load self, hence bypasses need to load language model
    def convColsToStmts(df_path, cols: list[str], col_out: str = "Processed_ents"):
        # Converts JSON string items in cols to JSON string in format of statements used by graph_builder
        root_name = os.path.splitext(df_path)[0] # Store root name
        df = importData(df_path)
        df_stmts = DataFrame()
        for ind, row in df.iterrows():
            ents_dict: dict[str, list[str]] = {c: json.loads(row[c]) for c in cols}
            ents_dict_fmt = json.dumps([ents_dict]) # Wrap in list to keep consistent formatting (GPT3 output has multiple output statements)
            new_entry = DataFrame({col_out: [ents_dict_fmt]})
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_stmts = pd.concat([df_stmts, new_entry])
            
        df_merged = pd.concat([df, df_stmts], axis=1)
        df_merged.to_csv(F"{root_name}_entsF.csv")
        LOG.info(F"Exported to {root_name}_entsF.csv")
        
        
        
    def resetDocs(self):
        self.doclist = []
    
    def lemmatizeCorpus(self,
                      df_path: Union[str, bytes, os.PathLike],
                      col: str,
                      pos_tags: list[str] = ["NOUN", "ADJ", "VERB", "ADV"],
                      stopwords: list[str] = [],
                      ) -> list[str]:
        """Imports XLS/CSV with texts in a specified column, processes, 
        and returns texts in a list with every token in their lemmatized form.

        Args:
            df_path (Union[str, bytes, os.PathLike]): Source path
            col (str): Column that contains texts to be lemmatized
            pos_tags (list, optional): Tags to be considered for lemmatization Defaults to ["NOUN", "ADJ", "VERB", "ADV"].

        Returns:
            list: list of strings of lemmatized texts
        """
        df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Screen for duplicates and presence of text for the column containing the text
        texts_lemmatized = []
        for index, row in df.iterrows():
            LOG.info(F"Lemmatizing row: {index}")
            if type(row[col]) == str:
                doc = self.NLP(row[col])
                row_lemmas = []
                for token in doc:
                    if token.pos_ in pos_tags and token.lemma_ not in stopwords and token.text not in stopwords:
                        row_lemmas.append(token.lemma_)
                row_lemmatized = " ".join(row_lemmas)
                texts_lemmatized.append(row_lemmatized)
        return texts_lemmatized
    
    # Start of single doc operations 
    
    def lemmatizeDoc(self, text: str,
                     pos_tags: list[str] = ["NOUN", "ADJ", "VERB", "ADV"], 
                     stopwords: list[str] = [],
                     ) -> str:
        # Takes document in form of string and returns all tokens (filtered by POS tags)
        # in their lemmatized form
        if isinstance(text, str): # Only parse if doc is string
            doc = self.NLP(text)
            doc_lemmas = []
            for token in doc:
                if all([token.pos_ in pos_tags,
                       token.lemma_ not in stopwords,
                       token.text not in stopwords]):
                    doc_lemmas.append(token.lemma_)
            doc_lemmatized = " ".join(doc_lemmas) # Join all lemmatized tokens
            return doc_lemmatized
        else:
            return ""
        
    def listSimilar(self, word: str, top_n: int = 10):
        """Prints top n similar words based on word vectors in the loaded model

        Args:
            word (str): Word to look up
            top_n (int, optional): How many similar words to print. Defaults to 10.
        """
        word_vector = np.asarray([self.NLP.vocab.vectors[self.NLP.vocab.strings[word]]])
        similar_vectors = self.NLP.vocab.vectors.most_similar(word_vector, n=top_n)
        similar_words = [self.NLP.vocab.strings[i] for i in similar_vectors[0][0]]
        LOG.info(similar_words)

def anonPath(text_path):
    
    text_root = os.path.splitext(text_path)[0]
    
    with open(text_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        text = "".join(lines)
    
    text_processed = anonText(text)
    
    text_out_path = F"{text_root}_anon.txt"
    with open(text_out_path, "w+") as file:
        file.write(text_processed)
        LOG.info(F"Wrote file to {text_out_path}")
    

def anonText(text):
    repl_name = "Smith"
    doc = NLP(text)
    
    text_processed = text # Copy str
    for ent in reversed(doc.ents): # Start at end so modifications don't shift frame
        if ent.label_ in ["PERSON"] and \
            not any(ex in ent.text.lower() for ex in ["guha", "achille", "hoffman"]): 
            text_processed = text_processed[:ent.start_char] + repl_name + text_processed[ent.end_char:]
    return text_processed
    
#%%
