import numpy as np
import pandas as pd
from collections import Counter

class Data_Handler(object):

    def __init__(self, filepath):
        self.full = pd.read_csv(filepath)[['description', 'variety']].dropna()
        self.freq_dict = self.create_freq_dict()
    
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

if __name__ == '__main__':
    '''
    Included here for diagnostics during development, otherwise usage
    will be primarily in other .py files and pipelines.
    '''

    wrangler = Data_Handler('data/cleaned_data.csv')