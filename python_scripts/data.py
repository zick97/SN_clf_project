import requests
import tarfile

import os, sys
from os.path import exists

from python_scripts.feature_extraction import *

# Function to download and extract the dataset
def download_dataset(url='', filename='', folder=''):
    # Download the dataset
    if exists(filename):
        print('Dataset already downloaded.')
    else:
        response = requests.get(url)
        if response.status_code == 200:
            # Save the downloaded file as dataset.tar.gz
            with open('dataset.tar.gz', 'wb') as file:
                file.write(response.content)
            print('Dataset downloaded successfully.')
        else:
            print('Failed to download the dataset.')

    # Extract the dataset
    if exists(folder):
        print('Dataset already extracted.')
    else:
        with tarfile.open('dataset.tar.gz', 'r:gz') as tar:
            tar.extractall()
        print('Dataset extracted successfully.')

#------------------------------------------------------------------------
import pandas as pd

# Function to create a dataframe from a .DAT file
def make_df(file_path):
    # Create an empty dataframe
    df = pd.DataFrame()
    # Open the file and read the content
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.split('\n')
    content = [line.split() for line in content]

    # Remove empty lines and lines starting with '#'
    content = [line for line in content if line != [] and line[0] != '#']

    # Insert the values from the .dat file into the dataframe
    for line in content:
        # Stop when reaching the 'NOBS' keyword
        if line[0] == 'NOBS:':
            break
        # Add the 'REDSHIFT_SPEC' keyword to the dataframe
        if line[0] == 'REDSHIFT_SPEC:':
            df[line[0]+'_ERROR'] = [float(line[3])]
        # Do the same for the 'HOST_GALAXY_PHOTO-Z' keyword
        elif line[0] == 'HOST_GALAXY_PHOTO-Z:':
            df[line[0]+'_ERROR'] = [float(line[3])]
        # Add the other keywords to the dataframe
        df[line[0]] = [line[1]]
    
    # Insert the remaining lines as lists of values associated to each filter
    # Get the VARLIST array
    varlist = [line for line in content if line[0] == 'VARLIST:'][0][1:]
    
    # Go through the OBS: lines
    obs_dict = dict()
    for line in content:
        if line[0] == 'OBS:':
            # Create a dictionary with the keywords as keys and the values as values
            temp_dict = dict(zip(varlist, line[1:]))
            # Merge this dictionary with the obs_dict dictionary, such that the keys are the keywords and 
            # the values are lists of values
            obs_dict = {key: obs_dict.get(key, []) + [value] for key, value in temp_dict.items()}

    # Group the values by the FLT keyword
    obs_df = pd.DataFrame(obs_dict).groupby('FLT').agg(list).drop('FIELD', axis=1)
    # Create a new column named VALUES, which contains the values of the other columns as a list of lists
    obs_df['VALUES'] = obs_df.values.tolist()
    obs_df.drop(obs_df.columns[:-1], axis=1, inplace=True)
    # Transpose the dataframe
    obs_df = (obs_df.T).reset_index(drop=True)
    # Merge this dataframe with the df dataframe
    df = pd.concat([df, obs_df], axis=1)

    return content, df

#------------------------------------------------------------------------
from tqdm import tqdm
import numpy as np

# Function to transform any strings (or lists of strings) to float (or lists of floats)
def str_to_float(lst):
    try:
        return float(lst)
    except:
        return np.array([np.array([float(value) for value in sublst]) for sublst in lst])

# Function to generate the dataframe for all the .DAT files inside a folder
# Notice that the function does not use the concat() method, which is very slow, but instead uses the loc() method
# in order to fill the dataframe step by step
def make_data(folder_path):
    # Get the dataset from the file 'dataset.csv'
    if exists('dataset.pkl'):
        print('Dataset already generated.')
        df = pd.read_pickle('dataset.pkl')
    else:
        print('Generating dataset...')
        # Get the list of .DAT files starting with the prefix 'DES_SN'
        file_list = [file for file in os.listdir(folder_path) if file.startswith('DES_SN') and file.endswith('.DAT')]

        # Create a dataframe with the index being the file names and the columns being the keywords
        df = pd.DataFrame(index=file_list, columns=make_df(os.path.join(folder_path, file_list[0]))[1].columns)

        # Iterate over the files using tqdm to show a progress bar
        for file in tqdm(file_list, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            file_path = os.path.join(folder_path, file)
            # Call the make_df() function to generate the dataframe for each file
            _, file_df = make_df(file_path)
            # Fill the dataframe with the values of the file_df dataframe
            df.loc[file] = file_df.loc[0]

        # Drop the IAUC column
        df.drop('IAUC:', axis=1, inplace=True)
        # Reset the index and delete the ':' character in the columns
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.str.replace(':', '')
        
        # Convert the numerical values of the dataframe to float, including the lists of values
        num_columns = ['RA', 'DECL', 'SNID', 'FAKE', 'MWEBV', 'REDSHIFT_SPEC_ERROR', 'REDSHIFT_SPEC', 
                    'HOST_GALAXY_GALID', 'HOST_GALAXY_PHOTO-Z', 'HOST_GALAXY_PHOTO-Z_ERROR', 'g', 'i', 'r', 'z']
        for column in num_columns:
            df[column] = df[column].apply(str_to_float)
        
        # Drop nan values from the dataframe and reset the index
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Modify the third array inside each filter's column of the dataframe so that each element is subtracted by the minimum value 
        # of the array - get the days since the first observation
        for flt in tqdm(['g', 'r', 'i', 'z'], desc='Filter\n', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            for i in range(len(df[flt])):
                df[flt][i][0] = np.array([x - df[flt][i][0][0] for x in df[flt][i][0]])
        
        # Save the dataframe as a .pkl file to preserve the lists of floats
        df.to_pickle('dataset.pkl')

    return df

#------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Function to plot each filter's flux with respect to T_obs, visualizing 4 light curves for the same SN
def plot_data(data, SNID):
    line = data[data['SNID'] == SNID].reset_index()
    # Get the total number of observation days
    line
    # Create a figure and an axis
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10,6))
    plt.suptitle(f'Light curve for SNID = {int(SNID)}', fontsize=18)
    fig.tight_layout()
    for k, flt in enumerate(['g', 'r', 'i', 'z']):
        x_values = line[flt][0][0]
        y_values = line[flt][0][1]
        error_values = line[flt][0][2]
        # Get the subplot indexes
        i = k // 2
        j = k % 2
        # Plot the data with error bars
        ax[i, j].errorbar(x_values, y_values, yerr=error_values, fmt='o', capsize=3.5, markersize=3.5, 
                          elinewidth=1, color='black')
        # Position the y-axis label
        if j == 0:
            ax[i, j].set_ylabel('$Flux$ $\\left[ 10^{-0.4*mag + 11} \\right]$', fontsize=13, rotation=90, labelpad=10)
        # Adjust other parameters
        ax[i, j].tick_params(axis='x')
        ax[i, j].grid(True, alpha=0.3, linestyle='--')
        ax[i, j].set_title(f'Filter ${flt}$', fontsize=14)
    plt.xlabel('$T_{obs}$ $\\left[ days \\right]$', fontsize=13, loc='right')
    plt.show()

#------------------------------------------------------------------------
# Function to plot both the fitted and the original light curves for each filter
def plot_light_curves(data, par_cols, SNID):
    line = data[data['SNID'] == SNID].reset_index()
    # Create a figure and an axis
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10,6))
    plt.suptitle(f'Light curve for SNID = {int(SNID)}', fontsize=18)
    fig.tight_layout()
    for k, flt in enumerate(['g', 'r', 'i', 'z']):
        x_values = line[flt][0][0]
        y_values = line[flt][0][1]
        error_values = line[flt][0][2]
        # Explicitly get the parameters of each light curve, otherwise the exec() function will not work
        A = line[f'A_{flt}'][0]
        B = line[f'B_{flt}'][0]
        t0 = line[f't0_{flt}'][0]
        t1 = line[f't1_{flt}'][0]
        Tr = line[f'Tr_{flt}'][0]
        Tf = line[f'Tf_{flt}'][0]
        # Get the subplot indexes
        i = k // 2
        j = k % 2
        # Plot the data with error bars
        ax[i, j].errorbar(x_values, y_values, yerr=error_values, fmt='o', capsize=3.5, markersize=3.5, 
                          elinewidth=1, color='black', label='Data')
        # Plot the fitted curve
        if A != np.nan:
            linspace = np.linspace(x_values[0], x_values[-1], 1000)
            ax[i, j].plot(linspace, fluxFunc(linspace, A, B, t0, t1, Tr, Tf), label='Fitted curve', color='red')
            ax[i, j].legend(loc='upper right', fontsize=7)
        # Position the y-axis label
        if j == 0:
            ax[i, j].set_ylabel('$Flux$ $\\left[ 10^{-0.4*mag + 11} \\right]$', fontsize=13, rotation=90, labelpad=10)
        # Adjust other parameters
        ax[i, j].tick_params(axis='x')
        ax[i, j].grid(True, alpha=0.3, linestyle='--')
        ax[i, j].set_title(f'Filter ${flt}$', fontsize=14)