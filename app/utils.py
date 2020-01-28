import sys
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords 
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import ne_chunk
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from spacy import displacy
from collections import Counter


def tokenize(text):
    '''
    Tokenize text.
    
    INPUT
        text (str): text to be tokenized
    OUTPUT
        tokens (list): list of tokens
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = nltk.word_tokenize(text)
    
    # Remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    # Reduce words to their root forms
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in words]
    
    return tokens


# Create custom transformer
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            if len(tokenize(sentence))>0:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    return 1
        return 0

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X):
        X_tagged = pd.Series(X, name='starting_verb').apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# Create custom transformer
entity_list = ['TIME', 'PERSON', 'GPE', 'DATE', 'NORP', 'MONEY', 'ORG', 'QUANTITY', 'CARDINAL', 'PERCENT', 'LOC', 'PRODUCT', 'FAC', 'ORDINAL', 'WORK_OF_ART', 'LANGUAGE', 'LAW', 'EVENT'] # https://spacy.io/api/annotation#named-entities
nlp = spacy.load('en_core_web_lg')
class NamedEntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def named_entities(self, text):
        doc = nlp(text)
        labels = [ent.label_ for ent in doc.ents]
        labels_count = Counter(labels)
        for ent in entity_list:
            if ent not in labels_count.keys():
                labels_count[ent] = 0
        return labels_count

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X):
        X_ent = X.apply(self.named_entities).apply(pd.Series)
        X_ent = X_ent.fillna(0)
        X_ent = X_ent[entity_list]
        return X_ent