import pickle
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import decomposition

data = pd.read_pickle("bm_words.pickle")

data.drop_duplicates(subset=['band', 'words_stemmed'], inplace=True)

data = data.groupby('band')['words_stemmed'] \
            .apply(lambda x: ','.join(x.astype(str))) \
            .reset_index()

# Transformation en vecteur td_idf
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(data['words_stemmed']).toarray()

df = pd.DataFrame(vector)
df.index = data['band']
df.to_csv("vector.csv", sep=',')