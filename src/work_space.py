import numpy as np
import pandas as pd
from data_handler import Data_Handler
from sklearn.feature_extraction.text import TfidfVectorizer
from eda_tools import top_x_words, graph_top_num, word_cloud
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.corpus import stopwords

def model_testing(model, X, y):
    '''
    Takes an unfit model as well as vectorized training and test data for
    k-fold cross validation testing.

    Parameters
    ----------
    model - 

    k - The number of k-folds to be used.

    X, y - Data and targets for use in the testing.

    Returns
    ----------
    None
    '''
    mod = model()
    scores = cross_val_score(mod, X, y)
    print(np.mean(scores), scores)

if __name__ == '__main__':
    #MODEL TESTING
    wrangler = Data_Handler('data/cleaned_data.csv')
    stops = wrangler.stop_words
    tfidf = TfidfVectorizer(stop_words=stops)

    df = wrangler.get_certain_varieties(['Chardonnay', 'Sauvignon Blanc'])
    y = df['variety']
    X = df['description']
    tfidf = TfidfVectorizer(stop_words=stops)
    X = tfidf.fit_transform(X)
    model_testing(ComplementNB, X, y)