# import "Pandas" library/package (and use shorthand "pd" for the package)
# Pandas provides functions for loading (storing) data from (to) files
import time

from numpy.core._multiarray_umath import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from plotchecker import ScatterPlotChecker
from matplotlib import pyplot as plt


def GetFeaturesLabels(m: object = 10, n: object = 10) -> object:
    house_dataset = load_boston()
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)

    x1 = house['RM'].values.reshape(-1, 1)  # vector whose entries are the average room numbers for each sold houses
    x2 = house['NOX'].values.reshape(-1, 1)  # vector whose entries are the nitric oxides concentration for sold houses

    x1 = x1[0:m]
    x2 = x2[0:m]

    np.random.seed(30)
    X = np.hstack((x1, x2, np.random.randn(m, n)))
    X = X[:, 0:n]
    # print('printing X...', X)
    y = house_dataset.target.reshape(-1, 1)  # creates a vector whose entries are the labels for each sold house
    y = y[0:m]

    return X, y


# X, y = GetFeaturesLabels()

# X, y = GetFeaturesLabels(10, 10)
# fig, axs = plt.subplots(1, 1, figsize=(15, 5))
# axs[0].scatter(X[:, 0], y)
# axs[0].set_title('average number of rooms per dwelling vs. price')
# axs[0].set_xlabel(r'feature $x_{1}$')
# axs[0].set_ylabel('house price $y$')
# axs.scatter(X[:, 0], y)
# axs.set_title('average number of rooms per dwelling vs. price')
# axs.set_xlabel(r'feature $x_{1}$')
# axs.set_ylabel('house price $y$')
# # axs[1].scatter(X[:, 1], y)
# # axs[1].set_xlabel(r'feature $x_{2}$')
# # axs[1].set_title('nitric oxide level vs. price')
# # axs[1].set_ylabel('house price $y$')
# # plt.scatter(X[:,0],y)
# plt.show()

# fig, axes = plt.subplots(1, 1, figsize=(8, 4))
# axes.scatter(X[:, 3], y)
# axes.set_title('$x_{4}$ vs. price $y$')
# axes.set_xlabel(r'feature $x_{4}$')
# axes.set_ylabel('house price $y$')
# plt.show()


# reg = LinearRegression(fit_intercept=False)
# reg = reg.fit(X, y)  # find the optimal weight vector W_opt. optimal wight vector (result) is stored in .coef_ #
#
# training_error = mean_squared_error(y, reg.predict(X))
#
# display(Math(r'$\mathbf{w}_{\rm opt} ='))
# optimal_weight = reg.coef_
#
# optimal_weight = optimal_weight.reshape(-1, 1)
# print("optimal weight ", optimal_weight)

# print("\nThe resulting training error is ", training_error)

"############################################################################################333"

m = 10  # we use 10 data points of the house sales database
max_r = 10  # maximum number of features used

start_time = time.time()
print("Start time ", start_time)

X, y = GetFeaturesLabels(m, max_r)  # read in m data points using max_r features

linreg_time= np.zeros(max_r)  # vector for storing the exec. times of LinearRegresion.fit() for each r
linreg_error= np.zeros(max_r)  # vector for storing the training error of LinearRegresion.fit() for each r

#X,y=(GetFeaturesLabels(m, 1))
#print('X ', X)
#print('y ',y)

for i in range(max_r):
    r = i+1
    #print('Starting round ',r)
    start_time = time.time()
    print('Getting %d houses with %d features' %(m,r))
    X,y=GetFeaturesLabels(m,r)
    print("X-y",X,y)
    reg = LinearRegression(fit_intercept=False)
    reg = reg.fit(X, y)
    training_error = mean_squared_error(y, reg.predict(X))
    end_time = (time.time() - start_time) * 1000
    linreg_time[i] = end_time
    linreg_error[i] = training_error

print('linreg_time ', linreg_time)
print('linreg_error', linreg_error)

# YOUR CODE HERE
# raise NotImplementedError()

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

assert linreg_time.shape == (10, ), "'linreg_time' has wrong dimensions."
assert linreg_error.shape == (10, ), "'linreg_error' has wrong dimensions."
assert linreg_error[0] > 0.01*linreg_error[9], "training error for n=1 is too small "
print('sanity check tests passed!')

"############################################################################################"

# max_m = 10                        # maximum number of data points
# feat = 2
# X, y = GetFeaturesLabels(max_m, feat)      # read in max_m data points using n=2 features
# train_error = np.zeros(max_m)         # vector for storing the training error of LinearRegresion.fit() for each r
#
# for i in range (max_m):
#     print('X[%d]=' %i, X[i])
#
# for i in range (max_m):
#     r=i+1
#     #print('Starting round ',r)
#     start_time=time.time()
#     print('Getting %d houses with %d features' %(r,feat))
#     X, y = GetFeaturesLabels(i+1,feat)
#     reg = LinearRegression(fit_intercept=False)
#     reg = reg.fit(X, y)
#     #print('reg.predict ', reg.predict(X))
#     train_error[i] = mean_squared_error(y, reg.predict(X))
#     print('train error ', train_error)
#     end_time=(time.time() - start_time)*1000
#     #print(end_time)
#
#
#
# #print(train_error[2])
# plot_x = np.linspace(1, max_m, max_m, endpoint=True)
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
# axes.plot(plot_x, train_error, label='MSE', color='red')
# axes.set_xlabel('number of data points (sample size)')
# axes.set_ylabel('training error')
# axes.set_title('training error vs. number of data points')
# axes.legend()
# plt.tight_layout()
# plt.show()

