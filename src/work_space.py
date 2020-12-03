import numpy as np
import pandas as pd
from data_handler import Data_Handler
from sklearn.feature_extraction.text import TfidfVectorizer
from eda_tools import top_x_words, graph_top_num, word_cloud
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

if __name__ == '__main__':
    #TOP WORDS PER VARIETAL
    # wrangler = Data_Handler('data/cleaned_data.csv')
    # pn = wrangler.get_top_num(15)
    # stops = wrangler.stop_words
    # vecto = TfidfVectorizer(max_features=100, stop_words=stops)
    # vecto.fit(pn['description'])
    # X = vecto.transform(pn['description'])
    # model = MultinomialNB()
    # model.fit(X, pn['variety'])
    # top_x_words(vecto,model, 10)

    #MODEL TESTING
    # wrangler = Data_Handler('data/cleaned_data.csv')
    # stops = wrangler.stop_words
    # tfidf = TfidfVectorizer(stop_words=stops)

    # df = wrangler.get_top_num(10)
    # y = df['variety']
    # X = df['description']
    # tfidf = TfidfVectorizer(stop_words=stops)
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # tfidf.fit(X_train)
    # X_train, X_test = tfidf.transform(X_train), tfidf.transform(X_test)
    # bayes = ComplementNB()
    # bayes.fit(X_train, y_train)
    # print(bayes.score(X_train, y_train))
    # print(bayes.score(X_test, y_test))

    #WORD CLOUD GENERATION
    # wrangler = Data_Handler('data/cleaned_data.csv')
    # df = wrangler.full
    # text = ' '.join(review for review in df['description'])
    # stops = wrangler.stop_words
    # word_cloud(text, stops)

    pass