import time
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import numpy as np
from sklearn.datasets import load_boston
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math

m = 10                            # we use 10 data points of the house sales database
max_r = 10                        # maximum number of features used

X,y = GetFeaturesLabels(m,max_r)  # read in m data points using max_r features

linreg_time = np.zeros(max_r)     # vector for storing the exec. times of LinearRegresion.fit() for each r
linreg_error = np.zeros(max_r)    # vector for storing the training error of LinearRegresion.fit() for each r


# linreg_time = ...
# linreg_error = ...
# Hint: loop "r" times.


# YOUR CODE HERE
raise NotImplementedError()

plot_x = np.linspace(1, max_r, max_r, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
axes[1].plot(plot_x, linreg_time, label='time', color='green')
axes[0].set_xlabel('features')
axes[0].set_ylabel('empirical error')
axes[1].set_xlabel('features')
axes[1].set_ylabel('time (ms)')
axes[0].set_title('training error vs number of features')
axes[1].set_title('computation time vs number of features')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()
