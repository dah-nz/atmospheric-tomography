import matplotlib.pyplot as plt
import numpy as np

"""
Plot a line graph with error shown for the std dev

Inputs:
    axis        Graph axis
    x           Independent values
    y_data      Arrays of data for each x value
    **kwargs    Matplotlib settings for plotting data
"""
def plot_error(axs, x, y_data, color, **kwargs):
    y = np.array([np.mean(d) for d in y_data])
    y_std = np.array([np.std(d) for d in y_data])
    axs.plot(x, y, color=color, **kwargs)
    # axs.fill_between(x, (y-y_std), (y+y_std), color=color, alpha=.1)