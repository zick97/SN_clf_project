import numpy as np
import pandas as pd
# Suppress the warning about chained assignment
pd.options.mode.chained_assignment = None  # default = 'warn'

from os.path import exists
from tqdm import tqdm

from scipy.optimize import curve_fit

def fluxFunc(t, A, B, t0, t1, Tr, Tf):
    return A * (1. + B * (t - t1)**2) * (np.exp(-(t - t0) / Tf) / (1. + np.exp(-(t - t0) / Tr)))

# -----------------------------------------------------------------------
# Class built to fit the light curves using a theoretical model
# Remove warnings 
import warnings
warnings.filterwarnings('ignore')

class getParams:
    def __init__(self, df):
        self.df = df
     
    # Function to fit a single light curve, associated to a line in the dataframe
    def fitCurve(self, line, flt):
        array = line[flt]
        try:
            # Bound values to avoid the curve_fit function to go to infinity
            pars, _ = curve_fit(fluxFunc, array[0], array[1], sigma=array[2], absolute_sigma=True,
                                        bounds=([10e-5, 10e-5, 0., 0., 0., 0.], [1000., 100., 100., 100., 100., 100.]),
                                                maxfev=100000)
        # Fill the array with zeros if the curve_fit function does not converge
        except RuntimeError:
            empty_array = np.empty(6)
            empty_array[:] = np.nan
            pars = empty_array
        return pars
    
    # Function to fit all the light curves in the dataframe
    def fitData(self):
        # Create the columns for the light curve parameters
        flts = ['g', 'r', 'i', 'z']
        pars = ['A', 'B', 't0', 't1', 'Tr', 'Tf']
        par_cols = []
        for flt in flts:
            for par in pars:
                par_cols += [f'{par}_{flt}']

        # If the dataset has already been generated, just load it
        if exists('datasets/dataset_train.pkl'):
            print('Dataset already generated.')
            self.df = pd.read_pickle('datasets/dataset_train.pkl')
        # Otherwise, fit the light curves
        else:
            # Add the new columns to the dataframe
            self.df = self.df.reindex(self.df.columns.tolist() + par_cols, axis=1)

            for index in tqdm(range(self.df.shape[0]), desc='Fitting light curves', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                fit_values = np.array([])
                for i, flt in enumerate(flts):
                    fit_values = np.append(fit_values, self.fitCurve(self.df.iloc[index], flt))

                for i, par in enumerate(par_cols):
                    # Attention: to exchange the values in the dataframe, you need to use .iloc AFTER the column name
                    self.df[par].iloc[index] = fit_values[i]
            # Save the dataframe as a .pkl file to preserve the lists of floats
            self.df.to_pickle('datasets/dataset_train.pkl')
            print('Dataset generated.')

        # Save the parameters column names for plotting purposes
        self.par_cols = par_cols

        return self.df