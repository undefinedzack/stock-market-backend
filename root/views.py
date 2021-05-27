from django.shortcuts import render
from django.conf import settings
import os
import nltk
import tweepy

# Data Manipulation

import numpy as np
import pandas as pd
import re
import sklearn

# Preprocessing the input data

import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Creating ngrams and vectorizing the data

from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser

import pandas_datareader as pdr
from datetime import date
import datetime

# Tools for building a model

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------------------------------------------------
def clean(tweet: str) -> str:
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    pat3 = r'[^a-zA-Z]'
    combined_pat2 = r'|'.join((combined_pat, pat3))

    # removing HTML
    text = BeautifulSoup(tweet, "lxml").get_text()

    # remove non-letters
    letters_only = re.sub(combined_pat2, " ", text)

    # converting to lower-case
    lowercase_letters = letters_only.lower()

    return lowercase_letters


def preprocessRAASi(Open, Close, Volume, senti, std, min, max):
    feature = 3
    X = []
    for i in range(len(Open) - feature + 1):
        lst = []
        for j in range(i, i + feature):
            lst.append(Open[j])
        for p in range(i, i + feature):
            lst.append(Close[p])
        for l in range(i, i + feature):
            lst.append(Volume[l])
        for k in range(i, i + feature):
            lst.append(senti[k])
        for m in range(i, i + feature):
            lst.append(std[m])
        for n in range(i, i + feature):
            lst.append(min[n])
        for o in range(i, i + feature):
            lst.append(max[o])
        X.append(lst)
    return np.array(X)


##### LEMMATIZATION
def lemmatize(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # lemmatize
    lemmatized_tokens = list(map(lemmatizer.lemmatize, tokens))

    # remove stop words
    meaningful_words = list(filter(lambda x: x not in stop_words, lemmatized_tokens))

    tweets = [ps.stem(word) for word in meaningful_words]
    return tweets


###### ALL TOGETHER
def preprocess(tweet: str) -> list:
    # clean tweet
    clean_tweet = clean(tweet)

    # tokenize
    tokens = word_tokenize(clean_tweet)

    # lemmatize
    lemmaz = lemmatize(tokens)

    return lemmaz


###### CLEANING WHOLE DATA BY PROCESSING EACH TWEET ONE BY ONE
def get_clean_data(tweets):
    return np.array(list(map(preprocess, tweets)))


###### BUILDING TRIGRAMS MODEL
def build_trigrams_model(cleaned_data):
    # creating n grams
    bigrams = Phrases(sentences=cleaned_data)
    trigrams = Phrases(sentences=bigrams[cleaned_data])

    # creating trigram model
    embedding_vector_size = 256
    trigrams_model = Word2Vec(
        sentences=trigrams[bigrams[cleaned_data]],
        size=embedding_vector_size,
        min_count=3, window=5, workers=4)

    return trigrams_model


###### VECTORIZING DATA
def vectorize_data(data, vocab: dict) -> list:
    print('Vectorize sentences...')
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda tweet: list(map(keys.index, filter(filter_unknown, tweet)))
    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized


###### FINAL DATA WITH PADDING
def vectorised_padded_data(cleaned_data):
    bigrams = Phrases(sentences=cleaned_data)
    trigrams = Phrases(sentences=bigrams[cleaned_data])
    X_data = trigrams[bigrams[cleaned_data]]

    print('Convert sentences to sentences with ngrams... (done)')
    input_length = 150

    trigrams_model = build_trigrams_model(cleaned_data)
    X_pad = pad_sequences(
        sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab),
        maxlen=input_length,
        padding='post')
    return X_pad


###### CLUBBING VECTORIZATION AND PADDING FUCTION
def suitable_data(tweets):
    cleaned_data = get_clean_data(tweets)
    return vectorised_padded_data(cleaned_data)


def test(request):
    df = pd.read_csv('root/graph.csv')

    dates = df['date'].tolist()
    original = df['Original'].tolist()
    predicted = df['Predicted'].tolist()

    original = [item[1:len(item) - 1] for item in original]
    predicted = [item[1:len(item) - 1] for item in predicted]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    new_model = tf.keras.models.load_model('saved_model')

    df1 = pd.read_csv('root/one1.csv')
    df2 = pd.read_csv('root/two2.csv')

    #~~~~~~~~~~~~~~~~~~~~Twitter~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # api = tweepy.API(auth, wait_on_rate_limit=True)
    #
    # text_query = 'boeing'
    # max_tweets = 500
    #
    # # Creation of query method using parameters
    # tweets = list(
    #     tweepy.Cursor(api.search, q=text_query, since="2021-04-26", until="2021-04-28", lang="en").items(max_tweets))
    #
    # texts = []
    # times = []
    #
    # for tweet in tweets:
    #     texts.append(tweet.text)
    #     times.append(str(tweet.created_at)[:10])
    #
    # df1 = pd.DataFrame()
    #
    # df1['clean_text'] = texts
    # df1['created_date'] = times
    # df1 = pd.read_csv('root/fetchedTexts.csv')
    #
    # df1['clean_text'] = df1['texts']
    # df1['created_date'] = df1['times']

    # today = date.today()
    # prev = today - datetime.timedelta(days=2)
    # df2 = pdr.DataReader('BA', start=prev, end=today, data_source='yahoo')
    # df2 = df2.reset_index(drop=False)


    df1 = df1.iloc[:, 0:3]
    X_pad = suitable_data(df1['clean_text'])
    lis = np.where(X_pad > 10678)
    for i in range(0, len(lis[0])):
        X_pad[lis[0][i]][lis[1][i]] = 10678

    outputs = new_model.predict(x=X_pad)

    df1 = df1.assign(sentiments=outputs)

    df1['created_date'] = pd.to_datetime(df1['created_date'])
    df1 = df1.sort_values(by='created_date')
    df_median = df1.groupby(['created_date'], as_index=False).agg(
        {'sentiments': ['mean', 'std', 'median', 'min', 'max']})
    df_median.columns = df_median.columns.droplevel(0)
    df_median.columns = ['created_date', 'mean', 'std', 'median', 'min', 'max']

    # Extracting date from 'Time' Column for a new column 'created_date'
    df2['created_date'] = df2['Time'].transform(lambda x : x.split(' ')[0])

    # Sorting According to Created_date
    df2['created_date'] = pd.to_datetime(df2['created_date'])
    df2.sort_values(by='created_date')

    df2_open = df2[['created_date', 'Open']]
    df2_close = df2[['created_date', 'Close']]
    df2_volume = df2[['created_date', 'Volume']]

    df2_open = df2_open.groupby(['created_date']).first()
    df2_close = df2_close.groupby(['created_date']).last()
    df2_volume = df2_volume.groupby(['created_date']).sum()

    final = pd.merge(df2_open, df2_volume, on='created_date')
    final = pd.merge(final, df2_close, on='created_date')

    # Some dates are missing in close open dataset due to holiday..getting them in a list
    missing_dates = pd.date_range(start='2019-01-01', end='2019-12-31').difference(final.index).tolist()

    # Function to merge sentiments of a holiday with its previous day sentiment
    print('~~~~~~~~~~~~~~~~~~df1prev~~~~~~~~~~~~~~~~~~~~')
    print(df1)
    df1 = df1.groupby(['created_date'], as_index=False).mean()
    print('~~~~~~~~~~~~~~~~~~df1after~~~~~~~~~~~~~~~~~~~~')
    print(df1)

    def missDateMerge(df1):
        newSentiments = df1['sentiments'].tolist()
        for i in range(1, len(newSentiments)):
            lst = []
            j = i - 1
            if df1['created_date'][i] in missing_dates:
                # print(df1['created_date'][i])
                while df1['created_date'][j] in missing_dates:  # if previous day sentiments are also holidays
                    lst.append(newSentiments[j])
                    j = j - 1
                lst.append(newSentiments[j])
                # print(lst)
                newSentiments[j] = (sum(lst) + newSentiments[i]) / (
                        len(lst) + 1)  # taking avg of all holiday sentiments and asssigning it to previous day
        return newSentiments

    newSentiments = missDateMerge(df1)

    df1['NewSentiments'] = newSentiments

    print('~~~~~~~~~~~~~~~~~~FINAL~~~~~~~~~~~~~~~~~~~~')
    print(final)
    print('~~~~~~~~~~~~~~~~~~df1~~~~~~~~~~~~~~~~~~~~')
    print(df1)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    full_final = pd.merge(df1, final, on='created_date', how='inner')

    full_final['created_date'] = pd.to_datetime(full_final['created_date'])
    full_final = pd.merge(df_median, full_final, on='created_date', how='inner')
    ##################################RUN FROM HERE########################################

    # full_final=pd.read_csv("lastone.csv")

    # we need input as [ [Open(day1) , Open(day2) , Open(day3) , Sentiment(day1) , Sentiment(day2) , Sentiment(day3)] ,
    #                   [Open[day2] , Open[day3], Open[day4] , Sentiment[day2] , Sentiment[day3] , Sentiment[day4]] , ...]
    # and y as [Open(day4) , Open(day5) , ........]

    print(full_final)
    X_test = preprocessRAASi(full_final['Open'].tolist(), full_final['Close'].tolist(), full_final['Volume'].tolist(),
                             full_final['NewSentiments'].tolist(), full_final['std'].tolist(),
                             full_final['min'].tolist(),
                             full_final['max'].tolist())

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    print(X_test)
    X_test = sc.fit_transform(X_test)

    n_features = 1
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    my_model = tf.keras.models.load_model('saved_model')

    predicted_stock_price = my_model.predict(X_test)
    y = (predicted_stock_price * (np.array(full_final['Open']).max() - np.array(full_final['Open']).min())) + np.array(
        full_final['Open']).min()
    y = y[0][0]

    context = {
        'dates': dates,
        'original': original,
        'predicted': predicted,
        'predictedStockPrice': y,
        'volume': full_final['Volume'].iloc[-1],
        'todaysOpeningPrice': full_final['Open'].iloc[-1],
        'todaysClosingPrice': full_final['Close'].iloc[-1]
    }

    return render(request, 'root/test.html', context)
