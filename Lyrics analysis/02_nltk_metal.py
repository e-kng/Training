import pickle
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer

file = open("black_metal_lyris.pkl", "rb")
data = pickle.load(file)
file.close()

data = pd.DataFrame.from_dict(data, orient='index')
data.reset_index(inplace=True)
data.columns = ['lyrics', 'band']
data['lyrics'].str.lower()

# Tokenisation des phrases
tokenizer = nltk.RegexpTokenizer(r'\w+') # permet de récupérer que les caractère alphanumeriques
data['tokenized_lyrics'] = data.apply(lambda row: tokenizer.tokenize(row['lyrics']), axis=1)

data = data.tokenized_lyrics.apply(pd.Series) \
    .merge(data, left_index = True, right_index= True) \
    .drop(['tokenized_lyrics'], axis=1) \
    .melt(id_vars = ['band', 'lyrics'], value_name = 'words') \
    .drop(["variable", "lyrics"], axis = 1) \
    .dropna()

# Comptages de mots
word_count = data.groupby(['band']).count().sort_values(by='words', ascending=False)
word_unique_count = data.groupby(['band']).nunique()

# plt.figure(figsize=(6,3))
# plt.bar(word_count.index, word_count['words'], color='#949feb', label='words count')
# plt.bar(word_unique_count.index, word_unique_count['words'], color='#010f73', label='unique words count')
# plt.xticks(rotation=90)
# plt.legend()
# plt.show()

# Nettoyage des mots inutiles (fréquence élevée)

words_cleaning = data['words'].value_counts()
words_cleaning = words_cleaning[words_cleaning < 350]
words_cleaning = pd.DataFrame(words_cleaning)
words_cleaning.reset_index(inplace=True)
words_cleaning.columns = ['words', 'occurency']
words_cleaning['len'] = words_cleaning.apply(lambda row: len(row['words']), axis=1)
words_cleaning = words_cleaning[words_cleaning['len']>3]

data_clean = data[data['words'].isin(words_cleaning['words'])]

# Stemming (racinisation)

stemmer = SnowballStemmer("english")
data_clean['words_stemmed'] = data_clean.apply(lambda row: stemmer.stem(row['words']), axis=1)

data_clean.to_pickle('bm_words.pickle')
