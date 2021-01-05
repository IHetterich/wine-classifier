import numpy as np
import pandas as pd
import pickle


def recommend(model, vectorizer, desc):
    desc_vect = vectorizer.transform([desc])
    probs = model.predict_proba(desc_vect)[0]
    idx = np.argsort(probs)
    top_five_idx = idx[-1: -6: -1]
    top_five = model.classes_[top_five_idx]
    return top_five


if __name__ == '__main__':
    vec = open('pickles/text_vec.pkl','rb')
    vecto = pickle.load(vec)

    mod = open('pickles/model.pkl', 'rb')
    model = pickle.load(mod)

    desc = 'Notes of citrus, bright on the palette with acidity and minerality.'

    print(recommend(model, vecto, desc))