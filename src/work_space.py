import numpy as np
import pandas as pd

from data_handler import Data_Handler
from sklearn.feature_extraction.text import TfidfVectorizer
from eda_tools import top_x_words, graph_top_num, word_cloud
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold


if __name__ == '__main__':
    wrangler = Data_Handler('data/cleaned_data.csv')
    df = wrangler.get_top_num(15)
    stops = wrangler.stop_words

    X = df['description']
    y = df['variety']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    vecto = TfidfVectorizer(stop_words=stops)
    X_train = vecto.fit_transform(X_train)
    X_test = vecto.transform(X_test)
    model = ComplementNB()
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test[0])
    idx = np.argsort(probs)
    print(model.classes_[idx])