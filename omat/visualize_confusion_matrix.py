# This function is used to plot the confusion matrix and normalized confusion matrix
import itertools
from sklearn.metrics import confusion_matrix

#%matplotlib inline
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotchecker import ScatterPlotChecker
from unittest.mock  import patch
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


def feature_matrix():
    """
    Generate a feature matrix representing the chemical measurements of wine samples in the dataset.

    :return: array-like, shape=(m, n), feature-matrix with n features for each of m wine samples. """

    wine = datasets.load_wine()  # load the dataset into the variable 'wine'
    features = wine['data']  # read out the features of the wine samples and store in variable 'features'
    n = features.shape[1]  # set n to the number of colums in features
    m = features.shape[0]  # set m equal to the number of rows in features

    ### STUDENT TASK ###
    X = np.zeros((m, n))
    # YOUR CODE HERE
    # raise NotImplementedError()
    X = features

    return X

def labels():
    """
    :return: array-like, shape=(m, 1), label-vector
    """
    wine = datasets.load_wine()  # load the dataset into the variable 'wine'
    cat = wine['target']  # read out the categories (0,1 or 2) of wine samples and store in vector 'cat'
    m = cat.shape[0]  # set m equal to the number of rows in features
    y = np.zeros((m,1));  # initialize label vector with zero entries

    ### STUDENT TASK ###
    # YOUR CODE HERE
    #raise NotImplementedError()

    for i in range(m):
        if cat[i] == 0:
            y[i,:] = 1
        else:
            y[i,:] = 0

    return y


def visualize_cm(cm):
    """
    Function visualizes a confusion matrix with and without normalization
    """
    plt.rc('legend', fontsize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im1, ax=axes[0])
    classes = ['Class 0', 'Class 1', 'Class 2']
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    axes[0].set_xlabel('predicted label $\hat{y}$')
    axes[0].set_ylabel('true label $y$')
    axes[0].set_title(r'$\bf{Figure\ 6.}$Without normalization')

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im2, ax=axes[1])

    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1].text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    axes[1].set_xlabel('predicted label $\hat{y}$')
    axes[1].set_ylabel('true label $y$')
    axes[1].set_title(r'$\bf{Figure\ 7.}$Normalized')

    axes[0].set_ylim(-0.5, 2.5)
    axes[1].set_ylim(-0.5, 2.5)

    plt.tight_layout()
    plt.show()


y = labels()
print(y.shape)
X = feature_matrix()

logReg = LogisticRegression(random_state=0,C=1e6)
logReg_fit = logReg.fit(X, y)
y_pred = logReg.predict(X)
y_pred = y_pred.reshape(-1,1)

cm = confusion_matrix(y, y_pred)

visualize_cm(cm)
