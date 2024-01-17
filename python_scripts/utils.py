import numpy as np
import matplotlib.pyplot as plt

from python_scripts.feature_extraction import fluxFunc

# -----------------------------------------------------------------------
# Inspect the differences in the parametric model for the flux using different parameter combinations
# By default, the function will plot the optimal parameters for the SNID 530 and other 4 parameter combinations
# to highlight the differences after increasing or decreasing the parameters
def plotFunc(parameters_list0=[], parameters_list1=[], labels0=[], labels1=[]):
    # Set up the x-axis
    x = np.linspace(0, 60, 1000)
    # Set up different parameter combinations
    if not len(parameters_list0) and not len(parameters_list1):
        parameters_list0 = [
            (44.45, 0.15, 20.86, 19.55, 1.23, 2.13),
            (44.45, 0.25, 20.86, 19.55, 1.23, 2.13),
            (44.45, 0.15, 35.86, 34.55, 1.23, 2.13),
        ]

        parameters_list1 = [
            (44.45, 0.15, 20.86, 19.55, 1.23, 2.13),
            (44.45, 0.15, 20.86, 19.55, 1.03, 2.13),
            (44.45, 0.15, 20.86, 19.55, 1.23, 1.90)
        ]

        # Define the labels for the legend
        labels0 = ['Optimal',
                'Higher $B$',
                'Higher $t_0$ and $t_1$',
        ]

        labels1 = ['Optimal',
                'Lower $T_r$',
                'Lower $T_f$'
        ]

        colors = ['firebrick', 'mediumpurple', 'orange']

    # Plotting
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10,4))
    plt.suptitle('Multiple Curves with Different Parameter Combinations', fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Iterate over the parameter combinations and plot the flux
    for i, params in enumerate(parameters_list0):
        y = fluxFunc(x, *params)
        ax[0].plot(x, y, label=labels0[i], color=colors[i])
        ax[0].grid(True, alpha=0.3, linestyle='--')
        ax[0].legend()
        ax[0].set_xlabel('$T_{obs}$ $\\left[ days \\right]$', fontsize=13, loc='center')

    for i, params in enumerate(parameters_list1):
        y = fluxFunc(x, *params)
        ax[1].plot(x, y, label=labels1[i], color=colors[i])
        ax[1].grid(True, alpha=0.3, linestyle='--')
        ax[1].legend()
        ax[1].set_xlabel('$T_{obs}$ $\\left[ days \\right]$', fontsize=13, loc='center')
    
    ax[0].set_ylabel('$Flux$ $\\left[ 10^{-0.4*mag + 11} \\right]$', fontsize=13, rotation=90, loc='center')
    plt.show()

# -----------------------------------------------------------------------
# Extract cluster memberships and probabilities
def getProbabilities(model, df):
    # Calculate Mahalanobis distances
    cluster_memberships = model.predict_proba(df)

    # Create a DataFrame for cluster memberships
    df_copy = df.copy()
    df_copy['PROBABILITY'] = cluster_memberships[:, 0]

    return df_copy

# -----------------------------------------------------------------------
import tensorflow as tf
# Create a model with Batch Normalization and Dropout
from tensorflow import keras
from functools import partial

def createModel(want_bn=True, want_dropout=True, classes=6, input_shape=0):
    # Clear any session and set the random seed
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create a Sequential model (no need to flatten the input)
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # Add Batch Normalization before the activation function
    if want_bn: keras.layers.BatchNormalization(),
    # Add Dropout after the activation function
    if want_dropout: keras.layers.Dropout(rate=0.2),
    # Add a Dense hidden layer with 300 neurons and ReLU activation function
    model.add(keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)))

    if want_bn: keras.layers.BatchNormalization(),
    if want_dropout: keras.layers.Dropout(rate=0.2),
    # Add a Dense hidden layer with 300 neurons and ReLU activation function
    model.add(keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)))

    if want_bn: keras.layers.BatchNormalization(),
    if want_dropout: keras.layers.Dropout(rate=0.2),
    # Add a Dense hidden layer with 100 neurons and ReLU activation function
    model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)))

    if want_bn: keras.layers.BatchNormalization(),
    if want_dropout: keras.layers.Dropout(rate=0.2),
    if classes == 2:
        # Add a Dense output layer with 1 neuron and sigmoid activation function
        model.add(keras.layers.Dense(1, activation='sigmoid')) 
    else: 
        # Add a Dense output layer with *classes* neurons (one per class) and softmax activation function
        model.add(keras.layers.Dense(classes, activation='softmax'))

    return model

# -----------------------------------------------------------------------
from sklearn.metrics import precision_score, recall_score
from keras.callbacks import Callback

# Define a custom callback to calculate average precision and recall at the end of each epoch
class PrecisionRecallCallback(Callback):
    def __init__(self, validation_data, verbose=0):
        super().__init__()
        self.validation_data = validation_data
        self.precision_values = []
        self.recall_values = []
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        self.precision_values.append(precision)
        self.recall_values.append(recall)

        if self.verbose > 0:
            print(f'Epoch {epoch + 1}: Precision = {precision:.4f}, Recall = {recall:.4f}')

# -----------------------------------------------------------------------
# Plot the learning curves
def plotHistory(history, precision_recall_callback=None):
    # Create a subplot grid with 1 row and 2 columns, one containing the loss and the other containing the accuracy
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Calculate the number of epochs (the x-axis)
    epochs = range(0, history.epoch[-1] + 1)

    # Plot the training loss
    ax[0].plot(history.history['loss'], label='Training Loss', color='firebrick', marker='o', markersize=3)
    # Plot the validation loss
    ax[0].plot(history.history['val_loss'], label='Validation Loss', marker='o', markersize=2)
    # Set the title
    ax[0].set_title('Training and Validation Loss')
    # Set the legend and the grid
    ax[0].legend()
    ax[0].grid(True, alpha=0.3, linestyle='--')

    # Plot the training accuracy
    ax[1].plot(history.history['accuracy'], label='Training Accuracy', color='firebrick', marker='o', markersize=2)
    # Plot the validation accuracy
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', markersize=2)

    if precision_recall_callback is not None:
        try:
            # Plot precision and recall over epochs
            ax[1].plot(epochs, precision_recall_callback.precision_values, label='Validation Precision', 
                    marker='o', markersize=2)
            ax[1].plot(epochs, precision_recall_callback.recall_values, label='Validation Recall', 
                    marker='o', markersize=2, color='grey')
        except: pass

    # Set the title
    ax[1].set_title('Training and Validation Metrics')
    # Set the legend and the grid
    ax[1].legend()
    ax[1].grid(True, alpha=0.3, linestyle='--')