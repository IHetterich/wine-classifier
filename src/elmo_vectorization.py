import numpy as np
import pandas as pd
import pickle
import spacy
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_hub as hub
import time


from data_handler import Data_Handler


nlp = spacy.load('en', disable=['parser', 'ner'])

elmo = hub.load("https://tfhub.dev/google/elmo/3")

def lemmatization(texts):
    '''
    Takes in text to be vectorized and lemmatizes it for the next steps.
    '''

    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

def elmo_data_prep():
    '''
    Takes in data to be vectorized and goes through a cleaning and
    lemmatization process so it plays licely with ELMo.
    '''

    start = time.time()
    print(time.asctime(time.localtime(start)))
    wrangler = Data_Handler('data/cleaned_data.csv')
    df = wrangler.get_top_num(15)
    stops = wrangler.stop_words
    
    X = df['description']
    y = df['variety']
    
    # Scrubbing methods
    punctuation = ',.!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    df['description'] = df['description'].apply(lambda x: ''.join(ch for ch in str(x) if ch not in set(punctuation)))
    df['description'] = df['description'].str.lower()
    df['description'] = df['description'].str.replace("[0-9]", " ")
    df['description'] = df['description'].apply(lambda x:' '.join([word for word in x.split() if word not in stops]))
    df['description'] = lemmatization(df['description'])

    # Saves data to new .csv
    df.to_csv('data/elmo_prepped_data.csv')
    print(df.head())
    print(time.time() - start)


def elmo_vectors(x):
    '''
    Goes through prepped data and vectorizes it returning embeddings.
    '''
    
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