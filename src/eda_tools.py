import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from wordcloud import ImageColorGenerator, WordCloud

from data_handler import Data_Handler

plt.style.use('ggplot')


def top_x_words(num):
    '''
    Prints out the top num words most highly weighted by tf-idf for a certain
    number of varieties. Selection of varieties can be changed on line 23.
    '''

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
    '''
    Graphs the number of reviews for the top num varieties in the dataset.
    '''

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
    '''
    Creates a wordcloud for a given variety to be chosen on line 75.
    '''

    wrangler = Data_Handler('data/cleaned_data.csv')
    # df = wrangler.full
    df = wrangler.get_certain_varieties(['Chardonnay'])
    text = ' '.join(review for review in df['description'])
    stops = wrangler.stop_words
    wordcloud = WordCloud(stopwords=stops, background_color='white', 
        width=500, height=600, colormap='viridis').generate(text)
    plt.figure(figsize=(6, 6), dpi=250)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def confusion_matrix():
    '''
    Creates a full confusion matrix for the top 15 varieties and displays it.
    Currently changes to vectorizer and model must be done manually.
    '''

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

    class_sort = ['Pinot Noir', 'Cabernet Sauvignon', 'Red Blend', 
        'Bordeaux-style Red Blend', 'Syrah', 'Merlot', 'Zinfandel',
        'Sangiovese', 'Malbec', 'Nebbiolo', 'Rosé', 'Chardonnay',
        'Sauvignon Blanc', 'Riesling', 'White Blend']
    plot_confusion_matrix(model, X_test, y_test, normalize='true', 
        xticks_rotation='vertical', labels=class_sort, include_values=False)
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
    elif visual == 'conf_mat':
        confusion_matrix()
    else:
        print("Invalid visualization selection.")
