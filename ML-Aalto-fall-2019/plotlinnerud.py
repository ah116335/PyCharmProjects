import requests
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


def LoadData(filename):
    df = pd.read_csv(filename)
    X = df.values
    print(X)
    m = np.shape(df)[0]
    n = np.shape(df)[1]
    #print(m)
    # Print("#######################################")

    # raise NotImplementedError()

    return X, m, n


def ScatterPlots():
    """
    Plot the scatterplot of all the data, then plot the scatterplot of the 3 subsets,
    each one with a different color

    return: axes object used for testing, containing the 2 scatterplots.
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    data, _, _, = LoadData("Data.csv")  # load data from csv file

    colors = ['r', 'g', 'b']
    #print(data)
    axes[0].scatter(data[:, 0], data[:, 1], label='All data')
    axes[0].legend()
    axes[1].scatter(data[0:200, 0], data[0:200, 1], c=colors[0], label='first 200 data points')
    axes[1].scatter(data[200:400, 0], data[200:400, 1], c=colors[1], label='second 200 data points')
    axes[1].scatter(data[400:600, 0], data[400:600, 1], c=colors[2], label='third 200 data points')
    axes[1].legend()

    return axes


axes = ScatterPlots()
plt.show()
