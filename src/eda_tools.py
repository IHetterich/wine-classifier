import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from data_handler import Data_Handler
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
plt.style.use('ggplot')


def top_x_words(num):
    wrangler = Data_Handler('data/cleaned_data.csv')
    pn = wrangler.get_top_num(15)
    stops = wrangler.stop_words
    vectorizer = TfidfVectorizer(max_features=100, stop_words=stops)
    vectorizer.fit(pn['description'])
    X = vectorizer.transform(pn['description'])
    model = ComplementNB()
    model.fit(X, pn['variety'])

    feature_words = vectorizer.get_feature_names()
    target_names = model.classes_
    for var in range(len(target_names)):
        print(f"\nTarget: {var}, name: {target_names[var]}")
        log_prob = model.feature_log_prob_[var]
        i_topn = np.argsort(log_prob)[::-1][:num]
        features_topn = [feature_words[i] for i in i_topn]
        print(f"Top {num} tokens: ", features_topn)


def graph_top_num(num):
    data_handler = Data_Handler('data/cleaned_data.csv')
    varietals = list(data_handler.freq_dict.keys())
    counts = list(data_handler.freq_dict.values())
    sort_idx = np.argsort(counts)
    top_idx = sort_idx[-1:-(num+1):-1]

    top_varietals = [varietals[idx] for idx in top_idx]
    top_counts = [counts[idx] for idx in top_idx]

    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    x = list(range(num))
    ax.bar(x, top_counts, tick_label=top_varietals)
    ax.set_xlabel('Varieties', fontsize=20)
    ax.set_ylabel('Reviews', fontsize=20)
    ax.set_title(f'Top {num} Reviewed Varieties', fontsize=25)
    plt.xticks(rotation=65, fontsize=15)
    plt.tight_layout()
    plt.show()


def word_cloud():
    wrangler = Data_Handler('data/cleaned_data.csv')
    df = wrangler.full
    text = ' '.join(review for review in df['description'])
    stops = wrangler.stop_words
    wordcloud = WordCloud(stopwords=stops, background_color='white', 
        width=800, height=500).generate(text)
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Houses several EDA methods")
    parser.add_argument('-v', '--visual', help='cloud, top_words, top_var')
    parser.add_argument('-n', '--number', type=int, help='Number of variables \
        if applicable')
    args = parser.parse_args()
    visual, num = args.visual, args.number
    if visual == 'cloud':
        word_cloud()
    elif visual == 'top_words':
        top_x_words(num)
    elif visual == 'top_var':
        graph_top_num(num)
    else:
        print("Invalid visualization selection.")