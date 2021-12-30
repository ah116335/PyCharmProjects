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

"##########Student Task Varying Number of Features.###############################################################333"

# m = 10  # we use 10 data points of the house sales database
# max_r = 10  # maximum number of features used
#
# start_time = time.time()
# print("Start time ", start_time)
#
# X, y = GetFeaturesLabels(m, max_r)  # read in m data points using max_r features
#
# linreg_time: ndarray = np.zeros(max_r)  # vector for storing the exec. times of LinearRegresion.fit() for each r
# linreg_error: ndarray = np.zeros(max_r)  # vector for storing the training error of LinearRegresion.fit() for each r
#
# X,y=(GetFeaturesLabels(m, max_r))
# print('X ', X)
# print('y ',y)


# for i in range(max_r):
#     r = i+1
#     print('Starting round ',r)
#     start_time = time.time()
#     print('%d houses with %d features' %(m,r))
#     #print('X =',X)
#     #print('y =',y)
#     print('X _r = ', X[:,:(r)])
#     reg = LinearRegression(fit_intercept=False)
#     reg = reg.fit(X[:,:(r)], y)
#     reg_predict=reg.predict(X[:,:(r)])
#     #training_error = mean_squared_error(y, reg.predict(X))
#     training_error = mean_squared_error(y, reg_predict)
#     print('training error =', training_error)
#     end_time = (time.time() - start_time) * 1000
#     linreg_error[i] = training_error
#     linreg_time[i] = end_time
#
# print('linreg_time ', linreg_time)
# print('linreg_error', linreg_error)
#
# # YOUR CODE HERE
# # raise NotImplementedError()
#
# plot_x = np.linspace(1, max_r, max_r, endpoint=True)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
# axes[1].plot(plot_x, linreg_time, label='time', color='green')
# axes[0].set_xlabel('features')
# axes[0].set_ylabel('empirical error')
# axes[1].set_xlabel('features')
# axes[1].set_ylabel('time (ms)')
# axes[0].set_title('training error vs number of features')
# axes[1].set_title('computation time vs number of features')
# axes[0].legend()
# axes[1].legend()
# plt.tight_layout()
# plt.show()
#
# assert linreg_time.shape == (10, ), "'linreg_time' has wrong dimensions."
# assert linreg_error.shape == (10, ), "'linreg_error' has wrong dimensions."
# assert linreg_error[0] > 0.01*linreg_error[9], "training error for n=1 is too small "
# print('sanity check tests passed!')

"############Student Task Varying Number of Data Points.############################################################"

max_m = 10                            # maximum number of data points
X, y = GetFeaturesLabels(max_m, 2)      # read in max_m data points using n=2 features
train_error = np.zeros(max_m)         # vector for storing the training error of LinearRegresion.fit() for each r
print('X ', X)

# train_error = ...
# Hint: loop "max_m" times.


# YOUR CODE HERE
# raise NotImplementedError()

for i in range (max_m):
    r=i+1
    print(' %d houses with %d features' %(r,2))
    X, y = GetFeaturesLabels(r,2)
    reg = LinearRegression(fit_intercept=False)
    reg = reg.fit(X, y)
    train_error[i] = mean_squared_error(y, reg.predict(X))
    #end_time=(time.time() - start_time)*1000

print(train_error[2])
plot_x = np.linspace(1, max_m, max_m, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
axes.plot(plot_x, train_error, label='MSE', color='red')
axes.set_xlabel('number of data points (sample size)')
axes.set_ylabel('training error')
axes.set_title('training error vs. number of data points')
axes.legend()
plt.tight_layout()
plt.show()

"############################################################################################"
#
# from sklearn import linear_model
#
# X,y = GetFeaturesLabels(10,1)   # read in 10 data points with single feature x_1 and label y
#
#
# ### fit a linear model to the clean data
# reg = linear_model.LinearRegression(fit_intercept=False)
# reg = reg.fit(X, y)
# y_pred = reg.predict(X)
#
# # now we intentionally perturb the label of the first data point
#
# y_perturbed = np.copy(y)
# y_perturbed[0] = 1000;
#
# ### fit a linear model to the perturbed data
#
# reg1 = linear_model.LinearRegression(fit_intercept=False)
# reg1 = reg1.fit(X, y_perturbed)
# y_pred_perturbed = reg1.predict(X)
#
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 4))
# axes[0].scatter(X, y, label='data')
# axes[0].plot(X, y_pred, color='green', label='Fitted model')
#
#
# # now add individual line for each error point
# axes[0].plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors') # add label to legend
# for i in range(len(X)-1):
#     lineXdata = (X[i+1], X[i+1]) # same X
#     lineYdata = (y[i+1], y_pred[i+1]) # different Y
#     axes[0].plot(lineXdata, lineYdata, color='red')
#
#
# axes[0].set_title('fitted model using clean data')
# axes[0].set_xlabel('feature x_1')
# axes[0].set_ylabel('house price y')
# axes[0].legend()
#
# axes[1].scatter(X, y_perturbed, label='data')
# axes[1].plot(X, y_pred_perturbed, color='green', label='Fitted model')
#
#
# # now add individual line for each error point
# axes[1].plot((X[0], X[0]), (y_perturbed[0], y_pred_perturbed[0]), color='red', label='errors') # add label to legend
# for i in range(len(X)-1):
#     lineXdata = (X[i+1], X[i+1]) # same X
#     lineYdata = (y_perturbed[i+1], y_pred_perturbed[i+1]) # different Y
#     axes[1].plot(lineXdata, lineYdata, color='red')
#
#
# axes[1].set_title('fitted model using perturbed training data')
# axes[1].set_xlabel('feature x_1')
# axes[1].set_ylabel('house price y')
# axes[1].legend()
#
# plt.show()
# plt.close('all') # clean up after using pyplot
#
# print("optimal weight w_opt by fitting to (training on) clean training data : ", reg.coef_)
# print("optimal weight w_opt by fitting to (training on) perturbed training data : ", reg1.coef_)

"############################################################################################"

# import numpy as np
# from matplotlib import pyplot as plt
#
# #------------------------------------------------------------
# # Define the Huber loss
# def Phi(t, c):
#     t = abs(t)
#     flag = (t > c)
#     return (~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t)
#
# #------------------------------------------------------------
# # Plot for several values of c
# fig = plt.figure(figsize=(10, 3.75))
# ax = fig.add_subplot(111)
#
# x = np.linspace(-5, 5, 100)
#
# for c in (1,2,10):
#     y = Phi(x, c)
#     ax.plot(x, y, '-k')
#
#     if c > 10:
#         s = r'\infty'
#     else:
#         s = str(c)
#
#     ax.text(x[6], y[6], '$c=%s$' % s,
#             ha='center', va='center',
#             bbox=dict(boxstyle='round', ec='k', fc='w'))
#
# ax.plot(x,np.square(x),label="squared loss")
#
# ax.set_xlabel(r'$y - \hat{y}$')
# ax.set_ylabel(r'loss $\mathcal{L}(y,\hat{y})$')
# ax.legend()
# plt.show()

"############################################################################################"
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor
#
# X,y = GetFeaturesLabels(10,1)   # read in 10 data points with single feature x_1 and label y
#
#
# ### fit a linear model (using Huber loss) to the clean data
#
# reg = HuberRegressor().fit(X, y)
# y_pred = reg.predict(X)
#
# # now we intentionaly perturb the label of the first data point
#
# y_perturbed = np.copy(y)
# y_perturbed[0] = 1000;
#
# ### fit a linear model (using Huber loss) to the perturbed data
#
# #reg1 = linear_model.LinearRegression(fit_intercept=False)
# reg1 = HuberRegressor().fit(X, y_perturbed)
# y_pred_perturbed = reg1.predict(X)
#
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 4))
# axes[0].scatter(X, y, label='data')
# axes[0].plot(X, y_pred, color='green', label='Fitted model')
#
#
# # now add individual line for each error point
# axes[0].plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors') # add label to legend
# for i in range(len(X)-1):
#     lineXdata = (X[i+1], X[i+1]) # same X
#     lineYdata = (y[i+1], y_pred[i+1]) # different Y
#     axes[0].plot(lineXdata, lineYdata, color='red')
#
#
# axes[0].set_title('fitted model using clean data')
# axes[0].set_xlabel('feature x_1')
# axes[0].set_ylabel('house price y')
# axes[0].legend()
#
# axes[1].scatter(X, y_perturbed, label='data')
# axes[1].plot(X, y_pred_perturbed, color='green', label='Fitted model')
#
#
# # now add individual line for each error point
# axes[1].plot((X[0], X[0]), (y_perturbed[0], y_pred_perturbed[0]), color='red', label='errors') # add label to legend
# for i in range(len(X)-1):
#     lineXdata = (X[i+1], X[i+1]) # same X
#     lineYdata = (y_perturbed[i+1], y_pred_perturbed[i+1]) # different Y
#     axes[1].plot(lineXdata, lineYdata, color='red')
#
#
# axes[1].set_title('fitted model using perturbed data')
# axes[1].set_xlabel('feature x_1')
# axes[1].set_ylabel('house price y')
# axes[1].legend()
#
# plt.show()
# plt.close('all') # clean up after using pyplot
#
# print("optimal weight w_opt by fitting on clean data : ", reg.coef_)
# print("optimal weight w_opt by fitting on perturbed data : ", reg1.coef_)

"############################################################################################"

# m = 10                            # we use 100 data points of the house sales database
# max_r = 10                        # maximum number of features used
#
# X,y = GetFeaturesLabels(m,max_r)  # read in 100 data points using 10 features
#
# linreg_time = np.zeros(max_r)     # vector for storing the exec. times of LinearRegresion.fit() for each r
# linreg_error = np.zeros(max_r)    # vector for storing the training error of LinearRegresion.fit() for each r
#
#
# for r in range(max_r):
#     reg_hub = HuberRegressor(fit_intercept=False)
#     start_time = time.time()
#     reg_hub = reg_hub.fit(X[:,:(r+1)], y)
#     end_time = (time.time() - start_time)*1000
#     linreg_time[r] = end_time
#     pred = reg_hub.predict(X[:,:(r+1)])
#     linreg_error[r] = mean_squared_error(y, pred)
#
# plot_x = np.linspace(1, max_r, max_r, endpoint=True)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
# axes[1].plot(plot_x, linreg_time, label='time', color='green')
# axes[0].set_xlabel('features')
# axes[0].set_ylabel('empirical error')
# axes[1].set_xlabel('features')
# axes[1].set_ylabel('Time (ms)')
# axes[0].set_title('training error vs number of features')
# axes[1].set_title('computation time vs number of features')
# axes[0].legend()
# axes[1].legend()
# plt.tight_layout()
# plt.show()