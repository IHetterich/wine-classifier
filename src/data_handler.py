import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords

class Data_Handler(object):

    def __init__(self, filepath):
        self.full = pd.read_csv(filepath)[['description', 'variety']].dropna()
        self.freq_dict = self.create_freq_dict()
        self.stop_words = self.generate_stop_words()
    
    def create_freq_dict(self):
        '''
        Creates a Counte dictionary of the varietals in the full dataset.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''

        freq_dict = Counter()
        freq_dict.update(self.full['variety'])
        return freq_dict
    
    def get_top_num(self, num):
        '''
        Returns a dataframe that only has datapoints for a certain number
        of the most frequent varietals in the full data.

        Parameters
        ----------
        num - The number of varietals to be examined.

        Returns 
        ----------
        A dataframe with datapoints for only the top num varietals.
        '''

        varietals = list(self.freq_dict.keys())
        counts = list(self.freq_dict.values())
        sort_idx = np.argsort(counts)
        top_idx = sort_idx[-1:-(num+1):-1]
        top_varietals = [varietals[idx] for idx in top_idx]
        return self.full[self.full['variety'].isin(top_varietals)]
    
    def get_certain_varieties(self, varieties):
        '''
        Takes a list of varietal names as they appear in the full data
        and returns a dataframe containing just those varietals.

        Parameters
        ----------
        varieties - The list of varietal names desired.

        Returns
        ----------
        A dataframe containg data for just the desired varieties.
        '''

        return self.full[self.full['variety'].isin(varieties)]

    def generate_stop_words(self):
        '''
        Generates a list of stop words for vectorization of data.
        Pulls words from the NLTK english stopword list and adds in
        all the varietals to avoid data leakage.

        Parameters
        ----------
        None

        Returns
        ----------
        A stopword list as described above.
        '''
        eng_stops = set(stopwords.words('english'))
        varieties = list(self.full['variety'].unique())
        varieties = set(' '.join(varieties).lower().replace('-',' ').split())
        customs = set(['chard', 'affile', 'alexandrie', 'aunis', 'avola', 'oeil', 'rondinella', 'st',
                        'wine', 'flavors', 'fruit'])
        return list(eng_stops.union(varieties).union(customs))

if __name__ == '__main__':
    '''
    Included here for diagnostics during development, otherwise usage
    will be primarily in other .py files and pipelines.
    '''

    wrangler = Data_Handler('data/cleaned_data.csv')