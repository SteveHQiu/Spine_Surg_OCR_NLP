#%%
import os, re, json, pickle
from dataclasses import dataclass
from typing import Union
from pathlib import Path

# Internals
from internals import LOG, importData
from nlp_spacy import SpacyModel

# Data science
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# NLP
from top2vec import Top2Vec
import gensim
from gensim.models import CoherenceModel, TfidfModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

#%%
STOPWORDS = []

#%%

class Clusterer:
    def __init__(self) -> None:
        self.file_path = ""
        self.root_name = ""
        self.root_base = ""
        self.df = DataFrame()
        self.col_corpora = ""
        
        self.vocab: gensim.corpora.Dictionary = None
        self.corpus_bow: list[list[tuple[int, int]]] = []
        self.model_lda: LdaModel = None
        
        self.model_vec: Top2Vec = None
        
        
    
    def importCorpora(self, df_path: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        self.file_path = df_path
        self.root_name = os.path.splitext(df_path)[0] # Get root name without extension 
        self.root_base = os.path.splitext(os.path.basename(df_path))[0] # Get base name without extension
        self.df = importData(df_path, screen_dupl=[col], screen_text=[col]) # Can't have any data type apart from str in list
        self.col_corpora = col
        LOG.info(F"Imported data from {df_path}")
        
    def genVocabAndBow(self, tfidf_thresh = 0.03, save = False):
        # Only for LDA pipeline, generates vocab including bi/trigrams and BOWs of every corpora
        # Uses TFD-IDF to process BOWs to remove irrelevant terms
        
        nlpmodel = SpacyModel(disable=["parser", "ner"])
        pos_tags = ["NOUN", "ADJ", "VERB", "ADV"]
        col_lemmatized = "Lemmatized"
        col_bow = "Bag_of_words"
        
        df_lemmatized = DataFrame()
        
        for ind, row in self.df.iterrows():
            text = row[self.col_corpora]
            doc_lemmatized = nlpmodel.lemmatizeDoc(text=text,
                                                   pos_tags=pos_tags,
                                                   stopwords=STOPWORDS)
            new_entry = DataFrame({col_lemmatized: [doc_lemmatized]})
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_lemmatized = pd.concat([df_lemmatized, new_entry])
            if ind % 10 == 0: # Log info every 10th document
                LOG.info(F"Lemmatized doc #{ind}")
        
        list_doc_lemmatized: list[str] = list(df_lemmatized[col_lemmatized]) # Convert into list for eventual usage in tfidf
        list_doc_lemmatized = [doc for doc in list_doc_lemmatized if doc] # Filter empty strings 
            
        list_doc_tokens = [gensim.utils.simple_preprocess(doc, deacc=True)
                    for doc in list_doc_lemmatized] # Is a list of lists of processed tokens
        
        # Bi/tri-grams
        bigram_phrases = gensim.models.Phrases(list_doc_tokens, min_count=5, threshold=50) 
        # Results in Phrases object whose index can be used to merge two tokens that are often found adjacent - hence bigrams
        bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases) 
        # Extracts the phraser portion of the Phrases oject for better performance https://radimrehurek.com/gensim/models/phrases.html
        corpus_bigrams = [bigram_phraser[doc] for doc in list_doc_tokens] # Combines tokens that fit bigram phrase, can be used as an intermediate step toward trigrams
        
        trigram_phrases = gensim.models.Phrases(bigram_phraser[list_doc_tokens], min_count=5, threshold=50)
        # Usess tokenized corpora, first merges bigrams and then uses output tokens to then detect trigrams
        trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
        # Same as bigram phraser, only using phraser portion of Phrases object
        
        corpus_bi_trigrams = [trigram_phraser[bigram_phraser[doc]] for doc in list_doc_tokens]
        # Mrges any bigrams and adjacent tokens into trigrams if possible to get both tri and bigrams
        # Can also go straight from single tokens into bi+trigrams by wrappering each doc in bigram phraser and then trigram phraser
        
        # TF-IDF 
        vocab = gensim.corpora.Dictionary(corpus_bi_trigrams) # Generates a vocabulary by mapping unique tokens in corpora to ID
        corpus_bow: list[list[tuple[int, int]]] = [vocab.doc2bow(text) for text in corpus_bi_trigrams] # Turn corpora into bag-of-words via vocab
        tfidf = TfidfModel(corpus_bow, id2word=vocab)
        
        
        
        low_value = tfidf_thresh
        
        df_bow = DataFrame()
        corpus_bow_processed = [] # Container for corpus bows after being processed by tfidf method 
        
        for ind, row in df_lemmatized.iterrows():
            doc_lemmatized: str = row[col_lemmatized]
            doc_tokens: list[str] = gensim.utils.simple_preprocess(doc_lemmatized, deacc=True)
            doc_bi_trigrams: list[str] = trigram_phraser[bigram_phraser[doc_tokens]]
            doc_bow: list[tuple[int, int]] = vocab.doc2bow(doc_bi_trigrams)
            low_value_words = [id for id, value in tfidf[doc_bow] if value < low_value]
            doc_bow_processed = [(id, count) for (id, count) in doc_bow 
                    if id not in low_value_words] # Remove low-value words
            corpus_bow_processed.append(doc_bow_processed)
            
            doc_bow_json = json.dumps(doc_bow_processed)
            new_entry = DataFrame({col_bow: [doc_bow_json]}) # Add BOW as json
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_bow = pd.concat([df_bow, new_entry])
        
        
        df_merged = pd.concat([self.df, df_lemmatized, df_bow], axis=1) # Concatenate all results
        df_merged.to_csv(f"{self.root_name}_bow.csv", index=False)
            
        self.vocab = vocab
        self.corpus_bow = corpus_bow_processed
        
        if save:
            vocab_bow_obj = (vocab, corpus_bow_processed)
            with open(F"{self.root_name}_vocab_bow.dat", "w+b") as file:
                pickle.dump(vocab_bow_obj, file)
                
            LOG.info(F"Saved vocab and bow tuple in {self.root_name}_vocab_bow.dat")
        
        

    def loadVocabBow(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Uses importCorpora to re-instantiate df info
        self.importCorpora(df_path=df_of_origin, col=col)
        with open(F"{self.root_name}_vocab_bow.dat", "rb") as file:
            vocab_bow_obj = pickle.load(file)
            
        self.vocab, self.corpus_bow = vocab_bow_obj # Unpack objects into class instance

    def clusterTopicsLda(self, num_topics: int, save = False):
        
        lda_model = LdaModel(corpus=self.corpus_bow,
                                id2word=self.vocab,
                                num_topics=num_topics,
                                random_state=100,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha="auto")
        
        self.model_lda = lda_model
        
        if save:
            lda_model.save(F"{self.root_name}_lda") # Creates this main file without an extension, has other files with extensions that is linked to this main file
            LOG.info(F"Saved LDA topic model to {self.root_name}_lda and associated files")
            
    def loadLdaModel(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Uses importVocabBow to instantiate df info and populate self.vocab and self.corpus_bow
        self.loadVocabBow(df_of_origin=df_of_origin, col=col)
        self.model_lda: LdaModel = LdaModel.load(F"{self.root_name}_lda")
        assert self.vocab == self.model_lda.id2word # Check that imported vocab is the same as one contained in model
        

    def visLdaTopics(self):
        num_topics = len(self.model_lda.get_topics())
        vis = pyLDAvis.gensim_models.prepare(topic_model=self.model_lda,
                                             corpus=self.corpus_bow,
                                             dictionary=self.vocab,
                                             R=30, # No. of terms to display in barcharts
                                             mds="mmds", # Function for calculating distances between topics
                                             sort_topics=False, # False to preserve original topic IDs
                                             )
        pyLDAvis.save_html(vis, f"figures/{self.root_base}_lda_n{num_topics}.html")
        LOG.info(f"Saved LDA visualization to figures/{self.root_base}_lda_n{num_topics}.html")
    
    def annotateCorporaLda(self):
        
        model = self.model_lda
        df_bow = importData(f"{self.root_name}_bow.csv")
        
        df_annot = DataFrame()
        
        for ind, row in df_bow.iterrows():
            doc_bow_json: str = row["Bag_of_words"]
            doc_bow: list[tuple[int, int]] = json.loads(doc_bow_json)
            topics: list[tuple[int, float]] = model.get_document_topics(doc_bow)
            topics.sort(key=lambda x: x[1], reverse=True) # Order using topic probability (float)
            top_topic = topics[0][0] # Get id of most likely topic
            new_entry = DataFrame({
                "Lda_topics": [topics],
                "Topic_lda": [top_topic],
            })
            new_entry.index = pd.RangeIndex(start=ind, stop=ind+1, step=1)
            df_annot = pd.concat([df_annot, new_entry])
            
        df_merged = pd.concat([df_bow, df_annot], axis=1)
        df_merged.to_csv(F"{self.root_name}_annotlda.csv")
        LOG.info(F"Successfully saved annotations to {self.root_name}_annotlda.csv")

        
    def clusterTopicsVec(self, save = False):
        corpus_df = self.df[self.col_corpora]
        corpus_list = list(corpus_df) # Need list format 
        self.model_vec = Top2Vec(corpus_list) # Need a minimum number of present topics to cluster, otherwise throws error
        num_topics = self.model_vec.get_num_topics()
        LOG.info(F"Successfully built topic model with {num_topics} topics")
        if save:
            self.model_vec.save(F"{self.root_name}_top2vec.dat")
            LOG.info(F"Saved trained model in {self.root_name}_top2vec.dat")
            
    def loadTop2VecModel(self, df_of_origin: Union[str, bytes, os.PathLike], col: str = "Abstract"):
        # Basically combines importCorpora with loading of corresponding top2vec model
        # Meant to be a starting point after cluterTopicsVec has been run 
        self.importCorpora(df_path=df_of_origin, col=col)
        self.model_vec = Top2Vec.load(F"{self.root_name}_top2vec.dat")
        LOG.info(F"Successfully loaded Top2Vec model from {self.root_name}_top2vec.dat")
    
    def genTopicWordClouds(self):
        # Only has support for top2vec models
        model = self.model_vec
        topic_size, topic_inds = model.get_topic_sizes()
        LOG.info(F"Found {len(topic_inds)} topics")
        figure_dir = F"figures/{self.root_base}_wordcloud"
        Path(figure_dir).mkdir(parents=True, exist_ok=True) # parents=True allows creation of intermediate parent directories
        LOG.info(F"Created directory for wordcloud figures at {figure_dir}")
        for size, topic in zip(topic_size, topic_inds):
            model.generate_topic_wordcloud(topic)
            plt.savefig(F"{figure_dir}/{self.root_base}_topic{topic}_size{size}.png",
                        bbox_inches="tight")
            
    def annotateCorporaVec(self, save = False):
        # Uses a generated or imported top2vec model to annotated topics to its original df
        # save -> save annotations byproduct in a separate container
        model = self.model_vec
        df_annot = DataFrame()
        topic_sizes, topic_nums = model.get_topic_sizes()
        for topic_size, topic_num in zip(topic_sizes, topic_nums): # Transform arrays into tuples
            documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_num, num_docs=topic_size)
            for doc, score, doc_id in zip(documents, document_scores, document_ids):
                entry = DataFrame({
                    self.col_corpora: [doc],
                    "Topic_vec": [topic_num],
                })
                df_annot = pd.concat([df_annot, entry])

        if save:
            df_annot.to_csv(F"{self.root_name}_t2v_annotations.csv", index=False)
            LOG.info(F"Exported annotations to {self.root_name}_t2v_annotations.csv")


        df_origin = self.df
        df_origin = df_origin.set_index(self.col_corpora) # Set index to col containing corpora
        df_annot = df_annot.set_index(self.col_corpora)
        
        # Merge abstract df with annotated df https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html#:~:text=Join%20DataFrames%20using%20their%20indexes.&text=If%20we%20want%20to%20join,have%20key%20as%20its%20index.&text=Another%20option%20to%20join%20using,to%20use%20the%20on%20parameter.
        df_merged = df_origin.join(df_annot, lsuffix='_left', rsuffix='_right')
        df_merged.to_csv(F"{self.root_name}_annotvec.csv")
        LOG.info(F"Successfully saved annotations to {self.root_name}_annotvec.csv")

