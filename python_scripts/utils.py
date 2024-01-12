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

    # Plotting
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10,4))
    plt.suptitle('Multiple Curves with Different Parameter Combinations', fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Iterate over the parameter combinations and plot the flux
    for i, params in enumerate(parameters_list0):
        y = fluxFunc(x, *params)
        ax[0].plot(x, y, label=labels0[i])
        ax[0].grid(True, alpha=0.3, linestyle='--')
        ax[0].legend()
        ax[0].set_xlabel('$T_{obs}$ $\\left[ days \\right]$', fontsize=13, loc='center')

    for i, params in enumerate(parameters_list1):
        y = fluxFunc(x, *params)
        ax[1].plot(x, y, label=labels1[i])
        ax[1].grid(True, alpha=0.3, linestyle='--')
        ax[1].legend()
        ax[1].set_xlabel('$T_{obs}$ $\\left[ days \\right]$', fontsize=13, loc='center')
    
    ax[0].set_ylabel('$Flux$ $\\left[ 10^{-0.4*mag + 11} \\right]$', fontsize=13, rotation=90, loc='center')
    plt.show()