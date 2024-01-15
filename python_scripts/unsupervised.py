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