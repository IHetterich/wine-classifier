import numpy as np
import pandas as pd
import argparse

def wheat_from_chaff(filepaths, save_point):
    '''
    Reads in a list of filepaths through pd.read_csv and then combines into a 
    single dataframe and removes duplicates based on description. Then saves
    resulting dataframe to the designated output filepath.

    Parameters
    ----------
    filepaths - A list or honestly any iterable of filepaths.

    Returns
    ----------
    None.
    '''

    dfs = []
    for file in filepaths:
        dfs.append(pd.read_csv(file))
    df_combo = pd.concat(dfs)
    df_no_dup = df_combo.drop_duplicates(subset='description')
    df_features = df_no_dup[['description', 'variety']]
    df_features.to_csv(save_point)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Takes in any number of \
        filepaths to .csv's and then saves a single duplicate .csv to a \
        specified filepath.")
    parser.add_argument('-i', '--inputs', action='append', 
        help='Input filepaths')
    parser.add_argument('-o', '--output', help='Output filepath')
    args = parser.parse_args()
    inputs, output = args.inputs, args.output
    wheat_from_chaff(inputs, output)