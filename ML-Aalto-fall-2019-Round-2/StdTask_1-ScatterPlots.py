import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import numpy as np
from sklearn.datasets import load_boston
import random

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
# axes.scatter ...
# YOUR CODE HERE
#raise NotImplementedError()
axs[0].scatter(X[:,3], y)
axes.set_title('$x_{4}$ vs. price $y$')
axes.set_xlabel(r'feature $x_{4}$')
axes.set_ylabel('house price $y$')

plt.show()

# plot tests

# the following two imports are for testing purposes only
# from plotchecker import ScatterPlotChecker
#
# X,y = GetFeaturesLabels(10,10)   # read in features and labels of house sales
#
# pc = ScatterPlotChecker(axes)
# np.testing.assert_array_equal(pc.x_data, X[:,3],f"The x values for the plot of the scatterplot are incorrect")
#
#
# print('sanity check tests passed!')