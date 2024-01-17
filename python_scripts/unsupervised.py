import numpy as np
import pandas as pd
# Suppress the warning about chained assignment
pd.options.mode.chained_assignment = None  # default = 'warn'

from os.path import exists
from tqdm import tqdm

# The following class implements the simplest version of PCA
class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # Center the data: it doesn't make any difference in case the data is already centered
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Calculate the explained variance ratio
        total_variance = np.sum(sorted_eigenvalues)
        explained_variance = sorted_eigenvalues[:self.n_components]
        self.explained_variance_ratio = explained_variance / total_variance

    def transform(self, X):
        # Center the data
        X_centered = X - self.mean

        # Project the data onto the principal components
        transformed_data = np.dot(X_centered, self.components)

        # Create a new dataframe with the transformed data
        transformed_df = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(self.n_components)])

        return transformed_df
    
from matplotlib import pyplot as plt
# -----------------------------------------------------------------------
# Plot the centroids of the clusters
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    # Plot the centroids as a white circle with a black cross inside it
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

# -----------------------------------------------------------------------
# Plot the decision boundaries derived from the KMeans model
def plot_decision_boundary(clusterer, df, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    # Define the range of the x and y axes
    mins = df.min(axis=0) - 0.1
    maxs = df.max(axis=0) + 0.1
    # Create a meshgrid - a set of points where we want to predict the cluster
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    # Predict the cluster for each point in the meshgrid
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Fill the decision boundaries
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    # Contour the decision boundaries
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    
    # Plot the data points
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=5, c='k', alpha=0.5)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.title('Decision Boundaries')

    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel(df.columns[0], fontsize=10)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel(df.columns[1], fontsize=10)
    else:
        plt.tick_params(labelleft=False)

# -----------------------------------------------------------------------
# Function that computes the NMI score after mapping the clusters
from itertools import permutations
from sklearn.metrics import normalized_mutual_info_score
import math

def nmiScore(clusterer, X_train, y_train):
    # Transform the predictions to a pandas DataFrame
    try:
        y_pred = pd.DataFrame(clusterer.predict(X_train.values), columns=['SNTYPE'])
    except AttributeError:
        y_pred = pd.DataFrame(clusterer.labels_, columns=['SNTYPE'])

    # Make a dictionary with index from 1 to 6 as keys and the original labels as values
    mapping_dict = {i+1: label for i, label in enumerate(y_train['SNTYPE'].unique())}

    # Generate all permutations of the keys
    all_permutations = permutations(mapping_dict.keys())

    max_nmi = 0.
    best_perm = None
    # Iterate through the permutations and create dictionaries
    print('Calculating the NMI score for all permutations...')
    for perm in all_permutations:
        # Create a copy of the predicted labels dataframe
        y_pred_copy = y_pred.copy()
        # Create a dictionary with the permutations
        perm_dict = {index: mapping_dict[key] for index, key in enumerate(perm, start=1)}
        # Map the predicted labels to the original labels
        y_pred_copy['SNTYPE'] = y_pred_copy['SNTYPE'].map(perm_dict)
        # Calculate the normalized mutual information score
        nmi = normalized_mutual_info_score(y_train['SNTYPE'], y_pred['SNTYPE'])
        if nmi > max_nmi:
            max_nmi = nmi
            best_perm = perm_dict
    
    return max_nmi, best_perm

# -----------------------------------------------------------------------
# Function that plots the decision boundaries of the GMM model
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, df, resolution=1000, show_ylabels=True, ax_names=['PC1', 'PC2']):
    # Plot the decision boundary
    mins = df.min(axis=0) - 0.1
    maxs = df.max(axis=0) + 0.1
    # Generating the grid
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    # Z is the positive log pdf, used to draw leveled colors
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 # levels' last item is the number of log levels to be shown
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    # Draw the borders (in red dashed line)
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(df[:, 0], df[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel(ax_names[0], fontsize=12)
    if show_ylabels:
        plt.ylabel(ax_names[1], fontsize=12)
    else:
        plt.tick_params(labelleft=False)

# -----------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Function that plots the confusion matrix
def plot_confusion_matrix(y_true, y_pred, size=(5, 5), want_report=False):
    # Create confusion matrix
    try:
        cm = confusion_matrix(y_true['SNTYPE'].values, y_pred['SNTYPE'].values)
    except KeyError:
        cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix as a heatmap using matplotlib
    plt.figure(figsize=size)
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Define the classes
    try:
        classes = np.unique(y_true['SNTYPE'].values)
    except KeyError:
        classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))

    # Place the labels on the axes
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

    plt.show()

    # Print the classification report
    if want_report:
        print('--------------------Classification Report--------------------')
        print(classification_report(y_true, y_pred))