import numpy as np
import pandas as pd
import pickle


def recommend(model, vectorizer, desc):
    '''
    Takes in a model, text vectorizer, and a single input. Returns five
    recommendations pulled from the top five most probable classes.

    Parameters
    ----------
    model - A classification model with a predict_proba method.

    vectorizer - A text vectorizer.

    desc - A single document to be vetorized and predicted.
    
    Returns
    ----------
    top_five - A list of the top five most likely classes for desc.
    '''

    desc_vect = vectorizer.transform([desc])
    probs = model.predict_proba(desc_vect)[0]
    idx = np.argsort(probs)
    top_five_idx = idx[-1: -6: -1]
    top_five = model.classes_[top_five_idx]
    return top_five


if __name__ == '__main__':
    '''
    Kept here for development and diagnostics. Method will otherwise
    be imported for use in other files.
    '''

    # vec = open('pickles/text_vec.pkl','rb')
    # vecto = pickle.load(vec)

    # mod = open('pickles/model.pkl', 'rb')
    # model = pickle.load(mod)

    # desc = 'Notes of citrus, bright on the palette with acidity and minerality.'

    # print(recommend(model, vecto, desc))