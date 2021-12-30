############################# IMPORTANT! #############################
# This cell needs to be run to load the necessary libraries and data #
######################################################################

# %matplotlib inline

from sklearn import datasets, tree
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotchecker import ScatterPlotChecker
from unittest.mock import patch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


def feature_matrix():
    """
    Generate a feature matrix representing the chemical measurements of wine samples in the dataset.

    :return: array-like, shape=(m, n), feature-matrix with n features for each of m wine samples. """

    wine = datasets.load_wine()  # load the dataset into the variable 'wine'
    features = wine['data']  # read out the features of the wine samples and store in variable 'features'
    m = features.shape[0]  # set m equal to the number of rows in features
    n = features.shape[1]  # set n to the number of colums in features

    ### STUDENT TASK ###
    X = np.zeros((m, n))

    X = features
    return X


def labels():
    """
    :return: array-like, shape=(m, 1), label-vector
    """
    wine = datasets.load_wine()  # load the dataset into the variable 'wine'
    cat = wine['target']  # read out the categories (0,1 or 2) of wine samples and store in vector 'cat'
    m = cat.shape[0]  # set m equal to the number of rows in features
    y = np.zeros((m, 1));  # initialize label vector with zero entries

    print('#############')
    for i in range(m):
        if cat[i] == 0:
            y[i] = 1
        else:
            y[i] = 0
    return y


# #print('data y = ')
# data = feature_matrix()
# #print('data = X')
# #print(data[2])
#
# wine = datasets.load_wine()
#
# X_data = wine['data']
# wine_class = wine['target']
# categories = wine_class.reshape(-1, 1)
# #print('X_data')
# #print(X_data[0])
# #print('wine_class')
# #print(wine_class)
# #print(categories)
#
# # print('data shape\t', X_data.shape, '\nlabels shape \t', categories.shape)
# # print("Number of samples from Class 0:", sum(wine_class == 0))
# # print("Number of samples from Class 1:", sum(wine_class == 1))
# # print("Number of samples from Class 2:", sum(wine_class == 2))
#
#
# data = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
# #data = features
# #print('features')
# #print(data)
# data['target'] = wine['target']
# data['class'] = data['target'].map(lambda ind: wine['target_names'][ind])
# #print(data.head(5))
#
# y = labels()
# X = feature_matrix()
# indx_1 = np.where(y == 1)[0] # index of each class 0 wine.
# indx_2 = np.where(y == 0)[0] # index of each not class 0 wine
# plt.rc('legend', fontsize=20)
# fig, axes = plt.subplots(figsize=(15, 5))
# #print("222")
# axes.scatter(X[indx_1, 0], X[indx_1, 1], c='g', marker ='x', label='y =1; Class 0 wine')
# axes.scatter(X[indx_2, 0], X[indx_2, 1], c='brown', marker ='o', label='y=0; Class 1 or Class 2 wine')
# axes.legend(loc='upper left')
# axes.set_xlabel('feature x1')
# axes.set_ylabel('feature x2')

#########Logistic loss##################################################

# def sigmoid_func(x):
#     f_x = 1 / (1 + np.exp(-x))
#     return f_x
#
#
# fig, axes = plt.subplots(1, 1, figsize=(15, 5))  # used only for testing purpose
#
# range_x = np.arange(-5, 5, 0.01).reshape(-1, 1)
# # print(range_x.shape)
# # print(len(range_x))
# # print(range_x)
# #print(range_x)
# #logloss_y1 = np.empty(len(range_x))
# logloss_y1 = np.zeros(len(range_x))
# logloss_y0 = np.empty(len(range_x))
# # print('####logloss_y1##################3')
# # print(logloss_y1)
# # print('####logloss_y0"#################3')
# # print(logloss_y0)
# # squaredloss_y1 = np.empty(len(range_x))
# # squaredloss_y0 = np.empty(len(range_x))
# plt.rc('legend', fontsize=20)
# plt.rc('axes', labelsize=20)
# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)
#
# for i in range(len(range_x)):
#     # print(range(len(range_x)))
#     print('i = %d' %i)
#     logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))  # logistic loss when true label y=1
#     logloss_y0[i] = -np.log(1 - sigmoid_func(range_x[i]))  # logistic loss when true label y=0
#
# # plot the results, using the plot function in matplotlib.pyplot.
#
# axes.plot(range_x, logloss_y1, linestyle=':', label=r'$y=1$', linewidth=5.0)
# axes.plot(range_x, logloss_y0, label=r'$y=0$', linewidth=5.0)
#
# axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
# axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
# axes.set_title("logistic loss", fontsize=20)
# axes.legend()
# plt.show()


##############Student Task. Logistic vs. Squared Error Loss.

def sigmoid_func(x):
    f_x = 1 / (1 + np.exp(-x))
    return f_x


# fig, axes = plt.subplots(1, 1, figsize=(15, 5))
#
# range_x = np.arange(-2, 2, 0.01).reshape(-1, 1)
# print(range_x.shape)
# logloss_y1 = np.empty(len(range_x))
# logloss_y0 = np.empty(len(range_x))
# squaredloss_y1 = np.empty(len(range_x))
# squaredloss_y0 = np.empty(len(range_x))
#
# plt.rc('legend', fontsize=20)
# plt.rc('axes', labelsize=40)
# plt.rc('xtick', labelsize=30)
# plt.rc('ytick', labelsize=30)
#
# for i in range(len(range_x)):
#     logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))  # logistic loss when true label y=1
#     logloss_y0[i] = -np.log(1 - sigmoid_func(range_x[i]))  # logistic loss when true label y=0
#     squaredloss_y1[i] = (1-range_x[i])**2
#     squaredloss_y0[i] = range_x[i]**2

### STUDENT TASK ###
# YOUR CODE HERE
# raise NotImplementedError()
# print(squaredloss_y0[0])
# print(squaredloss_y1[0])
# print(squaredloss_y0[-1])
# print(squaredloss_y1[-1])


# print(logloss_y1)
# plot the results, using the plot function in matplotlib.pyplot.

# IMPORTANT!: Please don't change below code for plotting, else the tests will fail and you will lose points.
#
# axes.plot(range_x, logloss_y1, linestyle=':', label=r'logistic loss $y=1$', linewidth=5.0)
# axes.plot(range_x, logloss_y0, label=r'logistic loss $y=0$', linewidth=5.0)
# axes.plot(range_x, squaredloss_y0 / 2, label=r'squared error for $y=0$', linewidth=5.0)
# axes.plot(range_x, squaredloss_y1 / 2, label=r'squared error for $y=1$', linewidth=5.0)
#
# axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
# axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
# axes.legend()
# plt.show()
#
# # Tests
#
# print('First entry of squaredloss_y0:', squaredloss_y0[0])
# print('First entry of squaredloss_y1:', squaredloss_y1[0])
# print('Last entry of squaredloss_y0:', squaredloss_y0[-1])
# print('Last entry of squaredloss_y1:', squaredloss_y1[-1])
#
#
# np.testing.assert_allclose(squaredloss_y0[0], 4.0, atol=1e-2, err_msg="First entry of squaredloss_y0 should be equal to approximately 4.0")
# np.testing.assert_allclose(squaredloss_y1[0], 9.0, atol=1e-2, err_msg="First entry of squaredloss_y1 should be equal to approximately 9.0")
# np.testing.assert_allclose(squaredloss_y0[-1], 3.96, atol=1e-2, err_msg="Last entry of squaredloss_y0 should be equal to approximately 3.96")
# np.testing.assert_allclose(squaredloss_y1[-1], 0.98, atol=1e-2, err_msg="Last entry of squaredloss_y1 should be equal to approximately 0.98")
#
# print('Sanity check tests passed!')


####################### Student Task. Logistic Regression.

# wine = datasets.load_wine()  # load wine datasets into variable "wine"
# X = wine['data']  # matrix containing the feature vectors of wine samples
# cat = wine['target'].reshape(-1, 1)  # vector with wine categories (0,1 or 2)
# m = cat.shape[0]  # set m equal to the number of rows in features
# y = np.zeros((m, 1));  # initialize label vector with zero entries
#
# for i in range(m):
#     if (cat[i] == 0):
#         y[i, :] = 1  # Class 0
#     else:
#         y[i, :] = 0  # Not class 0
#
# print('X.shape, y.shape')
# print(X.shape, y.shape)
# # Split X and y to training and test sets with parameters:test_size=0.2, random_state=0
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # print('X_train.shape, X_test.shape')
# # print(X_train.shape, X_test.shape)
# # print('y_train.shape, y_test.shape')
# # print(y_train.shape, y_test.shape)
#
# # initialize logistic regression
# logReg = LogisticRegression(random_state=0,C=1e6)
# print('2')
#
# # Train Logistic Regression Classifier
# logReg_fit = logReg.fit(X, y)
# print('3')
# # Predict the response for test dataset
# # y_pred = ...
# #y_pred = np.empty((m,1))
#
# y_pred = logReg.predict(X)
# print(y_pred.shape)
# y_pred = y_pred.reshape(-1,1)
# print(y_pred.shape)
# #print(y_pred)
#
#
# print('OK!')
# # YOUR CODE HERE
# #raise NotImplementedError()

################################# Demo. Linear Decision Boundary.

def calculate_accuracy(y, y_hat):
    """
    Calculate accuracy of your prediction

    :param y: array-like, shape=(m, 1), correct label vector
    :param y_hat: array-like, shape=(m, 1), label-vector prediction

    :return: scalar-like, percentual accuracy of your prediction
    """
    ### STUDENT TASK ###
    # YOUR CODE HERE
    correct_predictions = 0
    for i in range(m):
        if y_hat[i] == y[i]:
            correct_predictions = correct_predictions + 1

    return (correct_predictions / m) * 100


#    accuracy = metrics.accuracy_score(y, y_pred)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics

wine = datasets.load_wine()  # load wine datasets into variable "wine"
X = wine['data']  # matrix containing the feature vectors of wine samples
cat = wine['target'].reshape(-1, 1)  # vector with wine categories (0,1 or 2)

m = cat.shape[0]  # set m equal to the number of rows in features
y = np.zeros((m, 1));  # initialize label vector with zero entries

for i in range(m):
    if (cat[i] == 0):
        y[i, :] = 1  # Class 0
    else:
        y[i, :] = 0  # Not class 0
    # print(y[i,:])

logReg = LogisticRegression(random_state=0)
logReg = logReg.fit(X, y)
y_pred = logReg.predict(X).reshape(-1, 1)

# Tests
test_acc = calculate_accuracy(y, y_pred)

print('Accuracy of the result is: %f%%' % test_acc)

assert 80 < test_acc < 100, "Your accuracy should be above 80% and less than 100%"
assert test_acc < 99, "Your accuracy was too good. You are probably not using correct methods."

print('Sanity check tests passed!')

###################### Demo. Multiclass Classification.

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

wine = datasets.load_wine()  # load wine datasets into variable "wine"
X = wine['data']  # matrix containing the feature vectors of wine samples
y = wine['target'].reshape(-1, 1)  # vector with wine categories (0,1 or 2)

logReg = LogisticRegression(random_state=0, multi_class="ovr")  # set multi_class to one versus rest ('ovr')

logReg = logReg.fit(X, y)

y_pred = logReg.predict(X).reshape(-1, 1)

test_acc = calculate_accuracy(y, y_pred)
#
# print('Accuracy of the result is: %f%%' % test_acc)

###################### Demo. Confusion Matrix.


import itertools
from sklearn.metrics import confusion_matrix


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


# cm = ...
# YOUR CODE HERE
# raise NotImplementedError()

cm = confusion_matrix(y, y_pred)

# visualize_cm(cm)

################################ Student Task. Confidence in Classifications.

# make a prediction
# y_probs = ...
# YOUR CODE HERE
# raise NotImplementedError()
y_probs = logReg.predict_proba(X)

# show the inputs and predicted probabilities
# print('first five samples and their probabilities of belonging to classes 0, 1 and 2:')
for i in range(5):
    print("Probabilities of Sample", i + 1, ':', 'Class 0:', "{:.2f}".format(100 * y_probs[i][0], 2), '%', 'Class 1:',
          "{:.2f}".format(100 * y_probs[i][1]), '%', 'Class 2:', "{:.2f}".format(100 * y_probs[i][2]), '%')
#
# n_of_discarded_samples = 0

# YOUR CODE HERE
# raise NotImplementedError()
n_of_discarded_samples = 0

for i in range(len(y_probs)):
    # print(y_probs[i][0])
    if y_probs[i][0] < 0.9:
        n_of_discarded_samples = n_of_discarded_samples + 1
print('Number of discarded samples:', n_of_discarded_samples)

################################ SDemo. Decision Boundary of a Decision Tree..

# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    indx_1 = np.where(Y == 1)[0]  # index of each class 0 wine.
    indx_2 = np.where(Y == 0)[0]  # index of each not class 0 wine

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[indx_1, 0], X[indx_1, 1], marker='x', label='class 0', edgecolors='k')
    plt.scatter(X[indx_2, 0], X[indx_2, 1], marker='o', label='class 1', edgecolors='k')
    plt.xlabel(r'Feature 1')
    plt.ylabel(r'Feature 2')


# wine = datasets.load_wine()  # load wine datasets into variable "wine"
# X = wine['data'][:, :2]  # matrix containing the feature vectors of first 2 features of wine samples
# c = wine['target']  # vector contaiing the true categories as determined by human someliers
#
# m = cat.shape[0]  # set m equal to the number of rows in features
# y = np.zeros((m, 1));  # initialize label vector with zero entries
#
# for i in range(m):
#     if (c[i] == 0):
#         y[i] = 1  # Class 0
#     else:
#         y[i] = 0  # Not class 0
#
# tree = DecisionTreeClassifier()  # define object "tree" which represents a decision tree
# tree.fit(X, y)  # learn a decision tree that fits well the labeled wine samples
# y_pred = tree.predict(X)  # compute the predicted labels for the wine samples
#
# accuracy = metrics.accuracy_score(y, y_pred)  # compute the rate of correctly classified wine samples
# print("Accuracy:", round(100 * accuracy, 2), '%')
#
# plot_decision_boundary(tree, X, y)
# #plt.show()

###########################################Student Task. Decision Tree Classifier
# Load libraries
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

# load data to feature matrix X and label vector y
wine = datasets.load_wine()
X = wine['data']
y = wine['target'].reshape(-1, 1)
feature_cols = wine['feature_names']  # needed for visualization
print(feature_cols)

# Create Decision Tree classifer object with parameters: random_state=0, criterion='entropy'
# clf = ...
# Train Decision Tree Classifier
# clf_fit = ...
# Predict the response for test dataset
# y_pred = ...
# Use the metrics.accuracy_score function to calculate accuracy.
# accuracy = ...

# YOUR CODE HERE
# raise NotImplementedError()
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf_fit = clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = metrics.accuracy_score(y, y_pred)

print(y_pred)
# reshape y_pred to 2d matrix
y_pred = y_pred.reshape(-1, 1)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", round(100 * accuracy, 2), '%')

###########################  Student Task. Confusion Matrix.

# create a confusion matrix
# cm = confusion_matrix(...)

# YOUR CODE HERE
# raise NotImplementedError()
cm = confusion_matrix(y, y_pred)

# visualizing a confusion matrix
visualize_cm(cm)
