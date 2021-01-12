import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

# from data_handler import Data_Handler

class Recommender(object):
    '''
    A class to house the text vectorizer and stacked Naive Bayes/Random Forest 
    Classifiers that form the heart of this wine recommender.
    '''

    def __init__(self):
        self.nb = ComplementNB()
        self.rf = RandomForestClassifier()
        self.vecto = TfidfVectorizer()
    
    def _fit(self, data):
        '''
        Takes in the data for the recommender to be trained and fit to.

        Parameters
        ----------
        data - The filepath to the data being fit.

        Returns
        ----------
        None
        '''

        wrangler = Data_Handler(data)
        df = wrangler.get_top_num(15)
        X = df['description']
        y = df['variety']

        X = self.vecto.fit_transform(X)
        self.nb.fit(X, y)
        X = self.nb.predict_proba(X)

        self.rf.fit(X, y)

    def predict(self, text):
        '''
        Takes in a single input of tasting notes and runs it through our
        vectorizer and ensemble method to return the top five predicted
        varieties.

        Parameters
        ----------
        text - str - The input tastings notes.

        Returns
        ----------
        top_five - lst -  The top five predicted varieties for recommendation.
        '''

        vect = self.vecto.transform([text])
        probs = self.nb.predict_proba(vect)
        probs = self.rf.predict_proba(probs)[0]
        idx = np.argsort(probs)
        top_five_idx = idx[-1: -6: -1]
        top_five = self.rf.classes_[top_five_idx]
        return top_five

if __name__ == '__main__':
    '''
    This section exists primarily just for the pickling of a recommender
    otherwise it can be useful for testing and development.
    '''
    recommender = Recommender()
    recommender._fit('data/cleaned_data.csv')

    f = open('pickles/recommender.pkl', 'wb')
    pickle.dump(recommender, f)