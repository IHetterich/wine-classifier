import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def top_x_words(vectorizer, model, num):
    feature_words = vectorizer.get_feature_names()
    target_names = model.classes_
    for var in range(len(target_names)):
        print(f"\nTarget: {var}, name: {target_names[var]}")
        log_prob = model.feature_log_prob_[var]
        i_topn = np.argsort(log_prob)[::-1][:num]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {num} tokens: ", features_topn)

def graph_top_num(data_handler, num):
    varietals = list(data_handler.freq_dict.keys())
    counts = list(data_handler.freq_dict.values())
    sort_idx = np.argsort(counts)
    top_idx = sort_idx[-1:-(num+1):-1]

    top_varietals = [varietals[idx] for idx in top_idx]
    top_counts = [counts[idx] for idx in top_idx]

    fig, ax = plt.subplots(figsize=(15,15), dpi=100)
    x = list(range(num))
    ax.bar(x, top_counts, tick_label=top_varietals)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()