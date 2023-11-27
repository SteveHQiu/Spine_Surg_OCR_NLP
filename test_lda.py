#%%
from pathlib import Path
import pandas as pd
import top2vec
import matplotlib.pyplot as plt


from internals import LOG
df = pd.read_csv("temp_df.csv")

# df = pd.read_excel("tbi_ym.xlsx")
corpus_df = df["Abstract"]
corpus_list = list(corpus_df) # Need list format 
corpus_list = [a.replace("\n", "") for a in corpus_list]
model_vec = top2vec.Top2Vec(corpus_list, min_count=1, speed="deep-learn") # Need a minimum number of present topics to cluster, otherwise throws error
num_topics = model_vec.get_num_topics()

root_base = "test"
model = model_vec
topic_size, topic_inds = model.get_topic_sizes()
LOG.info(F"Found {len(topic_inds)} topics")
figure_dir = F"figures/{root_base}_wordcloud"
Path(figure_dir).mkdir(parents=True, exist_ok=True) # parents=True allows creation of intermediate parent directories
LOG.info(F"Created directory for wordcloud figures at {figure_dir}")
for size, topic in zip(topic_size, topic_inds):
    model.generate_topic_wordcloud(topic)
    plt.savefig(F"{figure_dir}/{root_base}_topic{topic}_size{size}.png",
                bbox_inches="tight")
#%%
import json, os
from typing import Union

import pandas as pd
from pandas import DataFrame

import gensim
from gensim.models import CoherenceModel, TfidfModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

from nlp_spacy import SpacyModel
from internals import LOG

from nltk.corpus import stopwords


STOPWORDS = stopwords.words('english')
EXCLUDE = ["leave", "patient", "no", "left", "right", "also", "today", "last",
             "year", "thank", "you", "involve", "say", "care", "time",
             "low", "back", "take", "report", "clinic", "commensurate", "history",
             "absent", "present", "symptom", "range_motion_normal", "tone_bulk_symmetrical",
             "other", "pain", "trial", "essentially", "up", "cigarette_day",
             "cm", "wait", "sense", "elsewhere", "inch_weigh", "habitus_endomorphic",
             "finding", "home", "pound", "kg", "here", "so", "sincerely", "assess",
             "follow", "see", "only", "reveal", "give", "feel", "such", "possibly",
             "twice", "child", "second", "discuss", "possible", "date", "endormorphic",
             "normal", "range_motion", "bid", "tid", "prn", "allergy", "qhs",
             "mg", "smoke", "stand_weigh_lbs", "positive", "negative", "reflex",
             "bilaterally"]

df = pd.read_csv("temp_df.csv")
self_col_corpora = "Abstract"
root_name = "prelim"
num_topics = 2

# Only for LDA pipeline, generates vocab including bi/trigrams and BOWs of every corpora
# Uses TFD-IDF to process BOWs to remove irrelevant terms
tfidf_thresh = 0.03

nlpmodel = SpacyModel(disable=["parser", "ner"])
pos_tags = ["NOUN", "ADJ", "VERB", "ADV"]
col_lemmatized = "Lemmatized"
col_bow = "Bag_of_words"

df_lemmatized = DataFrame()

for ind, row in df.iterrows():
    text = row[self_col_corpora]
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
corpus_processed = []
for corpus_doc in corpus_bi_trigrams:
    corpus_doc_mod = [w for w in corpus_doc if w not in EXCLUDE]
    corpus_processed.append(corpus_doc_mod)

# TF-IDF 
vocab = gensim.corpora.Dictionary(corpus_processed) # Generates a vocabulary by mapping unique tokens in corpora to ID
corpus_bow: list[list[tuple[int, int]]] = [vocab.doc2bow(text) for text in corpus_processed] # Turn corpora into bag-of-words via vocab
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


df_merged = pd.concat([df, df_lemmatized, df_bow], axis=1) # Concatenate all results
df_merged.to_csv(f"{root_name}_bow.csv", index=False)
    
self_vocab = vocab
self_corpus_bow = corpus_bow_processed

#%%
num_topics = 6
lda_model = LdaModel(corpus=self_corpus_bow,
                        id2word=self_vocab,
                        num_topics=num_topics,
                        random_state=100,
                        update_every=1,
                        chunksize=100,
                        passes=10,
                        alpha="auto")

self_model_lda = lda_model

num_topics = len(self_model_lda.get_topics())
vis = pyLDAvis.gensim_models.prepare(topic_model=self_model_lda,
                                        corpus=self_corpus_bow,
                                        dictionary=self_vocab,
                                        R=30, # No. of terms to display in barcharts
                                        mds="mmds", # Function for calculating distances between topics
                                        sort_topics=False, # False to preserve original topic IDs
                                        )
pyLDAvis.save_html(vis, f"figures/{root_name}_lda_n{num_topics}.html")
LOG.info(f"Saved LDA visualization to figures/{root_name}_lda_n{num_topics}.html")

# %%
