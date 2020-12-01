import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def top_x_words(vectorizer, model, num):
    feature_words = vectorizer.get_feature_names()
    target_names = model.classes_
    for cat in range(len(target_names)):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_prob = model.feature_log_prob_[cat]
        i_topn = np.argsort(log_prob)[::-1][:num]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {num} tokens: ", features_topn)