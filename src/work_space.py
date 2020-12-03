import numpy as np
import pandas as pd
from data_handler import Data_Handler
from sklearn.feature_extraction.text import TfidfVectorizer
from eda_tools import top_x_words, graph_top_num, word_cloud
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from nltk.corpus import stopwords

def model_testing(stops, X, y, max=None, ngrams=(1,1)):
    '''
    Takes an unfit model as well as vectorized training and test data for
    k-fold cross validation testing.

    Parameters
    ----------
    stops - Set of stop words.

    k - The number of k-folds to be used.

    X, y - Data and targets for use in the testing.

    Returns
    ----------
    None
    '''
    scores = []
    folder = StratifiedKFold()
    for train_idx, test_idx in folder.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        vecto = TfidfVectorizer(stop_words=stops, max_features=max, 
            ngram_range=ngrams)
        X_train = vecto.fit_transform(X_train)
        X_test = vecto.transform(X_test)
        model = ComplementNB()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    print(np.mean(scores), scores)

def vectorizer_hyper_test(params, stops, X, y):
    '''
    Takes in a dictionary of hyper-parameters much like a grid search and
    runs through all combinations testing them with model_testing.

    Parameters
    ----------
    params - Dictinary of hyper-parameters, keys are parameters, keys are values.

    stop - Set of stop words.

    X, y - Data and targets for testing.

    Returns
    ----------
    None
    '''

    for grams in params['n-grams']:
        for feats in params['max_features']:
            print(f'For {grams} and {feats}')
            model_testing(stops=stops, X=X, y=y, max=feats, ngrams=grams)

if __name__ == '__main__':
    params = {'max_features': [10000, 15000, 20000, 25000, 30000, None], 'n-grams': [(1,1)]}
    
    wrangler = Data_Handler('data/cleaned_data.csv')
    stops = wrangler.stop_words

    df = wrangler.get_top_num(15)
    y = df['variety'].to_numpy()
    X = df['description'].to_numpy()
    vectorizer_hyper_test(params, stops, X, y)