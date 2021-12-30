# import "Pandas" library/package (and use shorthand "pd" for the package)
# Pandas provides functions for loading (storing) data from (to) files
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import numpy as np
from sklearn.datasets import load_boston
import random


def GetFeaturesLabels(m=10, n=10):
    house_dataset = load_boston()
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
    print('#####')
    print('house')
    print(house)
    print('####')
    x1 = house['RM'].values.reshape(-1, 1)  # vector whose entries are the average room numbers for each sold houses
    x2 = house['NOX'].values.reshape(-1, 1)  # vector whose entries are the nitric oxides concentration for sold houses

    x1 = x1[0:m]
    x2 = x2[0:m]

    np.random.seed(30)
    X = np.hstack((x1, x2, np.random.randn(m, n)))

    X = X[:, 0:n]
    print("X  ######")
    print(X)
    y = house_dataset.target.reshape(-1, 1)  # creates a vector whose entries are the labels for each sold house
    y = y[0:m]
    print("y #################################")
    print(y)
    return X, y

X,y = GetFeaturesLabels()

