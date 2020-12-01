import numpy as np
import pandas as pd
import matplotlib as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from data_handler import Data_Handler
from sklearn.feature_extraction.text import CountVectorizer

def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. variance
    of the projections) for the principal components in a fit sklearn
    PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the scree plot.
      
    title: str
      A title for the scree plot.
    """
    num_components = pca.n_components
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
	    ax.annotate(r"{:2.2f}%".format(vals[i]), 
	     (ind[i]+0.2, vals[i]+0.005), 
		   va="bottom", 
		   ha="center", 
		   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
	    ax.set_title(title, fontsize=16)

def plot_mnist_embedding(ax, X, y, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.
      
    title: str
      A title for the plot.
    """
    ax.axis('off')
    ax.patch.set_visible(False)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    varieties = y.unique()
    for i, var in enumerate(varieties):
      temp = X[y == var]
      x = temp.iloc[:, [0]]
      y = temp.iloc[:, [1]]
      ax.scatter(x, y, color=colors[i], label=var)

if __name__ == '__main__':
  wrangler = Data_Handler('data/cleaned_data.csv')
  test = wrangler.get_top_num(2)
  X_test = test['description']
  y_test = test['variety']
  vecto = CountVectorizer(stop_words='english')
  X_vect = vecto.fit_transform(X_test)
  smol = TruncatedSVD(n_components=10)
  smol_test = smol.fit_transform(X_vect)
  fig, ax = plt.subplots()
  scree_plot(ax, smol)
  plt.show()