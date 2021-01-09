import numpy as np
import pandas as pd
import pickle
import time

from data_handler import Data_Handler
from eda_tools import top_x_words, graph_top_num, word_cloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB


def model_testing(stops, X, y, max=None, ngrams=(1,1)):
    '''
    Takes an unfit model as well as vectorized training and test data for
    k-fold cross validation testing.

    Parameters
    ----------
    stops - Set of stop words.

    X, y - Data and targets for use in the testing.

    Returns
    ----------
    None
    '''
    scores = []
    folder = StratifiedKFold()
    for train_idx, test_idx in folder.split(X, y):
        cycle = time.time()
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # vecto = TfidfVectorizer(stop_words=stops, max_features=max, 
        #     ngram_range=ngrams)
        # X_train = vecto.fit_transform(X_train)
        # X_test = vecto.transform(X_test)

        nb = RandomForestClassifier()
        nb.fit(X_train, y_train)
        X_train = nb.predict_proba(X_train)
        X_test = nb.predict_proba(X_test)

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        scores.append(rfc.score(X_test, y_test))
        print(time.time() - cycle)
    print(np.mean(scores), scores)


def vectorizer_hyper_test(params, stops, X, y):
    '''
    Takes in a dictionary of hyper-parameters much like a grid search and
    runs through all combinations testing them with model_testing.

    Parameters
    ----------
    params - Dictinary of hyper-parameters, keys are parameters, values are 
                lists of values to be tested.

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

    
def pickling():
    '''
    Creates and pickles both the vectorizer and model for use in prediction.

    Parameters
    ----------
    None

    Returns
    ----------
    None
    '''

    wrangler = Data_Handler('data/cleaned_data.csv')
    stops = wrangler.stop_words
    df = wrangler.get_top_num(15)
    X = df['description']
    y = df['variety']

    vecto = TfidfVectorizer(stop_words=stops)
    X = vecto.fit_transform(df['description'])
    f =  open('pickles/text_vec.pkl', 'wb')
    pickle.dump(vecto, f)

    model = ComplementNB()
    model.fit(X, y)
    m = open('pickles/model.pkl', 'wb')
    pickle.dump(model, m)

if __name__ == '__main__':
    '''
    Due to the nature of the functions here I haven't instantiated argparse.
    Honestly opening this file up and directly changing the parameter dictionary
    seems more straightforward and easier than having to pass in 10 values
    on the command line, especially when juggling multiple hyper parameters.

    Keeping whatever was found best in the last grid search as the only values.
    '''

    # params = {'max_features': [None], 'n-grams': [(1,1)]}
    
    wrangler = Data_Handler('data/cleaned_data.csv')
    stops = wrangler.stop_words
    df = wrangler.get_top_num(15)
    y = df['variety'].to_numpy()
    # X = df['description'].to_numpy()

    i = open('data/elmo_vectors.pkl', 'rb')
    elmo = pickle.load(i)
    # df = df + 2.5
    # print(df.shape)
    # print(np.mean(df))
    # print(np.max(df))
    # print(np.min(df))

    model_testing(stops, elmo, y)
    # vectorizer_hyper_test(params, stops, X, y)