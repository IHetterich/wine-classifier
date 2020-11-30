import numpy as np
import pandas as pd

if __name__ == '__main__':
    '''
    Gonna be honest, none of this is that much so we're just running with
    this little functional pipeline
    '''
    
    df1 = pd.read_csv('data/winemag-data-130k-v2.csv')
    df2 = pd.read_csv('data/winemag-data_first150k.csv')
    df_combo = pd.concat([df1, df2])
    df_no_dup = df_combo.drop_duplicates(subset='description')
    df_features = df_no_dup[['description', 'variety']]
    df_features.to_csv('data/cleaned_data.csv')