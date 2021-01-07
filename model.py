import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# NLP Packages
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter



import re
from bs4 import BeautifulSoup
import pickle

# Modeling packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


df = pd.read_csv('reviews.csv')

df['sentiment'] = np.where(df['Score']>3, 1, 0)
df = df[df['Score'] != 3]

df['Text'] = df['Text'].str.lower()

def remove_URL(text):
    """Remove URLs from a text string"""
    return re.sub(r"http\S+", "", text)

df['Text'] = df['Text'].apply(lambda x: remove_URL(x))

def getText(x):
    # soup = BeautifulSoup(x, 'lxml')
    soup = BeautifulSoup(x)
    text = soup.get_text()
    return text

df['Text'] = df['Text'].apply(lambda x: getText(x))

# removing special characters
def remove_spl(x):
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    return x

df['Text'] = df['Text'].apply(lambda x: remove_spl(x))

def remove_num_words(x):
    x = re.sub('\w*\d\w*', ' ', x)
    return x

df['Text'] = df['Text'].apply(lambda x: remove_num_words(x))

def remove_stop_words(x, stop_words):
    text = ' '.join(word for word in word_tokenize(x) if word not in stop_words)
    return text

df['Text'] = df['Text'].apply(lambda x: remove_stop_words(x, stop_words))

def cleanText(text):
    global stop_words
    text = text.lower()
    text = remove_URL(text)
    text = getText(text)
    text = remove_spl(text)
    text = remove_num_words(text)
    text = remove_stop_words(text, stop_words)
    return text

#BOW
bow_counts = CountVectorizer(tokenizer= word_tokenize,stop_words=stop_words, ngram_range=(1,1))

bow_data = bow_counts.fit_transform(df['Text'])

# Save the vectorizer
vec_file = 'bow_counts.pickle'
pickle.dump(bow_counts, open(vec_file, 'wb'))

#LRBOW
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data, df['sentiment'], test_size = 0.3, random_state = 42)
lr_bow = LogisticRegression()
lr_bow.fit(X_train_bow, y_train_bow)

test_pred_lr_bow = lr_bow.predict(X_test_bow)

# Save the model
mod_file = 'lr_bow.model'
pickle.dump(lr_bow, open(mod_file, 'wb'))

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_counts = TfidfVectorizer(tokenizer= word_tokenize,stop_words=stop_words, ngram_range=(1,1))

tfidf_data = tfidf_counts.fit_transform(df['Text'])

# Save the vectorizer
vec_file = 'tfidf_counts.pickle'
pickle.dump(tfidf_counts, open(vec_file, 'wb'))

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data, df['sentiment'], test_size = 0.3,random_state = 42)

lr_tf_idf = LogisticRegression()

lr_tf_idf.fit(X_train_tfidf,y_train_tfidf)

test_pred_lr_tf_idf = lr_tf_idf.predict(X_test_tfidf)

# Save the model
mod_file = 'lr_tf_idf.model'
pickle.dump(lr_tf_idf, open(mod_file, 'wb'))

def predictText(text):
    lrbow = -1
    lrtfidf = -1
    global stop_words

    text = cleanText(text)

    #load vectorizer
    loaded_bow = pickle.load(open('bow_counts.pickle', 'rb'))
    # load the model
    loaded_model_bow = pickle.load(open('lr_bow.model', 'rb'))
    # make a prediction
    lrbow = int(loaded_model_bow.predict(loaded_bow.transform([text])))

    # load vectorizer
    loaded_tfidf = pickle.load(open('tfidf_counts.pickle', 'rb'))
    # load the model
    loaded_model_tfidf = pickle.load(open('lr_tf_idf.model', 'rb'))

    # make a prediction
    lrtfidf = int(loaded_model_tfidf.predict(loaded_tfidf.transform([text])))

    if lrbow == 1:
        pred_bow = 'POSITIVE'
    else:
        pred_bow = 'NEGATIVE'
    
    if lrtfidf == 1:
        pred_tfidf = 'POSITIVE'
    else:
        pred_tfidf = 'NEGATIVE'

    return pred_bow, pred_tfidf

