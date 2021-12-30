# Import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# m = 20
# n = 10
#
# X = np.random.randn(m,n)   # create feature vectors using random numbers
# y = np.random.randn(m,1)   # create labels using random numbers
#
# print((X))
# print('#######')
# print(y)
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) # 80% training and 20% test
#
# plt.rc('legend', fontsize=20)
# plt.rc('axes', labelsize=20)
# fig1, axes1 = plt.subplots(figsize=(15, 5))
# axes1.scatter(X[:, 0], X[:, 1], c='g', s=200,marker ='x', label='original dataset')
# axes1.legend(loc='best')
# axes1.set_xlabel('feature x1')
# axes1.set_ylabel('feature x2')
#
# fig2, axes2 = plt.subplots(figsize=(15, 5))
# axes2.scatter(X_train[:, 0], X_train[:, 1], c='g', s=200,marker ='o', label='training set')
# axes2.scatter(X_val[:, 0], X_val[:, 1], c='brown', s=200,marker ='s', label='validation set')
# axes2.legend(loc='best')
# axes2.set_xlabel('feature x1')
# axes2.set_ylabel('feature x2')
#
# #fig2.show()


############################Demo loading the data

def GetFeaturesLabels(m=20, n=10):
    house_dataset = load_boston()  # load some house sales data
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
    x1 = house['RM'].values.reshape(-1, 1)  # vector whose entries are the average room numbers for each sold houses
    x2 = house['NOX'].values.reshape(-1, 1)  # vector whose entries are the nitric oxides concentration for sold houses
    #print('x1')
    #print(x1)
    # print('x2')
    # print(x2)
    x1 = x1[0:m]  # choose first feature of first m data points from the database
    x2 = x2[0:m]  # choose second feature of first m data pionts from the database
    #print('x1-')
    #print(x1)
    # print('x2')
    # print(x2)

    np.random.seed(15)
    X = np.hstack((x1, x2, np.random.randn(n, m).T))  # add some "extra" features, maybe it helps :-)

    # print('X.hstack')
    # print(X)
    X = X[:, 0:n]  # some reshaping of the numpy arrays
    # print('X..')
    # print(X)
    y = house_dataset.target.reshape(-1, 1)  # creates a vector whose entries are the labels for each sold house
    # print('y.')
    # print(y)
    y = y[0:m]  # chosse labels of first m data points in the database
    # print('y..')
    # print(y)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)  # normalize feature values to standard value range
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)

    return X, y

import time
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# m = 20                        # we use the first m=20 data points from the house sales database
# n = 10                        # maximum number of features used
#
# X,y = GetFeaturesLabels(m,n)  # read in m data points using n features
# linreg_error = np.zeros(n)    # vector for storing the training error of LinearRegresion.fit() for each r
# # print('X')
# # print(X)
# for r_minus_1 in range(n):  # loop over number of features r (minus 1)
#     #print(r_minus_1)
#     reg = LinearRegression(fit_intercept=False)   # create an object for linear predictors
#     reg = reg.fit(X[:,:(r_minus_1 + 1)], y)                 # find best linear predictor (minimize training error)
#     pred = reg.predict(X[:,:(r_minus_1 + 1)])               # compute predictions of best predictors
#     #print(pred)
#     linreg_error[r_minus_1] = mean_squared_error(y, pred) # compute training error
#     #print(linreg_error)
#
# plot_x = np.linspace(1, n, n, endpoint=True)      # plot_x contains grid points for x-axis
#
# # Plot training error E(r) as a function of feature number r
# plt.rc('legend', fontsize=12)
# plt.plot(plot_x, linreg_error, label='$E(r)$', color='red')
# plt.xlabel('# of features $r$')
# plt.ylabel('training error $E(r)$')
# plt.title('training error vs number of features')
# plt.legend()
#plt.show()


########################################## Student Task. Generate Training and Validation Set.

from sklearn.model_selection import train_test_split # Import train_test_split function

m = 20                        # we use the first m=20 data points from the house sales database
n = 10                        # maximum number of features used
# X = np.zeros((m,n))
# y = np.zeros((m,1))
X,y = GetFeaturesLabels(m,n)  # read in m data points using n features

### STUDENT TASK ###
# Compute the training and validation sets
# X_train, X_val, y_train, y_val = ...
# YOUR CODE HERE
#raise NotImplementedError()

# X_train
# X_val
# y_train
# y_val

# print('#### X')
# print(X)
# print('#### y')
# print(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) # 80% training and 20% test

# print('#### X_train')
# print(X_train)
# print('#### X_val')
# print(X_val)
#
# print('#### y_train')
# print(y_train)
# print('#### y_val')
# print(y_val)
#
# print('#### end1')
# print('#### end2')


# print(np.amin(y))

######################### Student Task. Compute Training and Validation Error.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

err_train = np.zeros([n, 1])
err_val = np.zeros([n, 1])

for r_minus_1 in range(n):  # loop over number of features r (minus 1)
    Reg_train = LinearRegression(fit_intercept=False)   # create an object for linear predictors
    #Reg_val=LinearRegression(fit_intercept=False)
    #print(X_train[:,:(r_minus_1 + 1)])
    print('1-1-1')
    print(r_minus_1)
    print('x-train')
    print(X_train[:,:(r_minus_1 + 1)])
    print('x-val')
    print(X_val[:, :(r_minus_1 + 1)])
    Reg_train_fit  = Reg_train.fit(X_train[:,:(r_minus_1 + 1)], y_train)
    pred_train = Reg_train_fit.predict(X_train[:,:(r_minus_1 + 1)])
    pred_val   = Reg_train_fit.predict(X_val[:, :(r_minus_1 + 1)])
    print('Reg_train_pred')
    print(pred_train)
    print('Pred-val')
    print(pred_val)
    err_train[r_minus_1] = mean_squared_error(y_train, pred_train)
    err_val[r_minus_1]   = mean_squared_error(y_val, pred_val)
    print('err_train')
    print(err_train)
    print('err_val')
    print(err_val)

#err_val=mean_squared_error(y_val,pred_train)

print('#### end3')
print('amin err_train')
print(np.amin(err_train))
print(np.argmin(err_train))
print('amin err_val')
print(np.amin(err_val))
print(np.argmin(err_val))
best_model = (np.argmin(err_val)+1)
print(best_model)