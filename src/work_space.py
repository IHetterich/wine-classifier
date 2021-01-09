'''
This file has just been a place that develop and test out various ideas and
functions, In the end anything that is actually deployed will be housed in an
actually formatted and well documented file elsewhere. I use this mainly as an
additional protection against breaking already working code that is being
regularly used
'''

import numpy as np
import pandas as pd
import pickle
# import spacy
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_hub as hub
import time


# from data_handler import Data_Handler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from eda_tools import top_x_words, graph_top_num, word_cloud
# from sklearn.naive_bayes import MultinomialNB, ComplementNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold


# nlp = spacy.load('en', disable=['parser', 'ner'])

elmo = hub.load("https://tfhub.dev/google/elmo/3")

def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

def elmo_data_prep():
    start = time.time()
    print(time.asctime(time.localtime(start)))
    wrangler = Data_Handler('data/cleaned_data.csv')
    df = wrangler.get_top_num(15)
    stops = wrangler.stop_words
    
    X = df['description']
    y = df['variety']
    
    # print(X.head())
    punctuation = ',.!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    df['description'] = df['description'].apply(lambda x: ''.join(ch for ch in str(x) if ch not in set(punctuation)))
    df['description'] = df['description'].str.lower()
    df['description'] = df['description'].str.replace("[0-9]", " ")
    df['description'] = df['description'].apply(lambda x:' '.join([word for word in x.split() if word not in stops]))
    # df['description'] = lemmatization(df['description'])

    df.to_csv('data/elmo_prepped_data.csv')
    print(df.head())
    print(time.time() - start)


def elmo_vectors(x):
    cycle = time.time()
    embeddings = elmo.signatures['default'](tf.convert_to_tensor(x))["default"]
    
    # sess = tf1.Session()
    # sess.run(tf1.global_variables_initializer())
    # sess.run(tf1.tables_initializer())
    # # return average of ELMo features
    # avg = sess.run(tf1.reduce_mean(embeddings,1))
    print(time.time() - cycle)
    return embeddings

if __name__ == '__main__':
    start = time.time()
    print(time.asctime(time.localtime(start)))

    df = pd.read_csv('data/elmo_prepped_data.csv')
    chunks = [df[i:i+100] for i in range(0, df.shape[0], 100)]
    elmo_vects = [elmo_vectors(x['description']) for x in chunks]
    elmo_final_vects = np.concatenate(elmo_vects, axis = 0)

    pickle_out = open('data/elmo_vectors.pkl','wb')
    pickle.dump(elmo_final_vects, pickle_out)
    pickle_out.close()
    print(time.time() - start)