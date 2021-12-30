def GetFeaturesLabels(m: object = 20, n: object = 10) -> object:
    house_dataset = load_boston()  # load some house sales data
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
    x1 = house['RM'].values.reshape(-1, 1)  # vector whose entries are the average room numbers for each sold houses
    x2 = house['NOX'].values.reshape(-1, 1)  # vector whose entries are the nitric oxides concentration for sold houses

    x1 = x1[0:m]  # choose first feature of first m data points from the database
    x2 = x2[0:m]  # choose second feature of first m data pionts from the database

    np.random.seed(15)
    X = np.hstack((x1, x2, np.random.randn(n, m).T))  # add some "extra" features, maybe it helps :-)
    X = X[:, 0:n]  # some reshaping of the numpy arrays
    y = house_dataset.target.reshape(-1, 1)  # creates a vector whose entries are the labels for each sold house
    y = y[0:m]  # chosse labels of first m data points in the database
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)  # normalize feature values to standard value range
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)

    return X, y

# Import KFold class from scikitlearn library
from sklearn.model_selection import KFold
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# m = 20  # we use the first m=20 data points from the house sales database
# n = 10  # maximum number of features used
# X = np.random.randn(m,n)   # create feature vectors using random numbers
# y = np.random.randn(m,1)   # create labels using random numbers
#
# K = 5  # number of folds/rounds/splits
# kf = KFold(n_splits=K, shuffle=False)
# kf = kf.split(X)
#
# kf = list(kf)  # kf is a list representing the rounds of k-fold CV
#
#
# X, y = GetFeaturesLabels(m, n)  # read in m data points with n features
# r = 2  # we use only first two features for linear predictors h(x) = w^{T}x
#
# train_errors_per_cv_iteration = []
# test_errors_per_cv_iteration = []
#
# # for loop over K rounds
#
# for train_indices, test_indices in kf:
#     reg = LinearRegression(fit_intercept=False)
#     reg = reg.fit(X[train_indices, :(r + 1)], y[train_indices])
#     y_pred_train = reg.predict(X[train_indices, :(r + 1)])
#     train_errors_per_cv_iteration.append(mean_squared_error(y[train_indices], y_pred_train))
#     y_pred_val = reg.predict(X[test_indices, :(r + 1)])
#     test_errors_per_cv_iteration.append(mean_squared_error(y[test_indices], y_pred_val))
#
# err_train = np.mean(train_errors_per_cv_iteration)  # compute the mean of round-wise training errors
# err_val = np.mean(test_errors_per_cv_iteration)  # compute the mean of round-wise validation errors

# print("Ttrain_errors_per_cv_iteration: " )
# print(train_errors_per_cv_iteration)
#
# print("Test_errors_per_cv_iteration:"  )
# print(test_errors_per_cv_iteration)
#
# print("Training error (averaged over 5 folds): ", err_train)
# print("Validation error (averaged over 5 folds):", err_val)
# print("###############################################################")

#########################33 Demo. Ridge Regression.

# m=20
# n=10
# XX,yy = GetFeaturesLabels(m,n)  # read in m data points using n features
# XX_train, XX_val, yy_train, yy_val = train_test_split(XX, yy, test_size=0.2, random_state=2)
#
# from sklearn.linear_model import Ridge
# alpha_scaled = 2*n
# ridge = Ridge(alpha=alpha_scaled, fit_intercept=True)
# ridge.fit(XX_train, yy_train)
# y_pred = ridge.predict(XX_train)
# w_opt = ridge.coef_
# err_train = mean_squared_error(y_pred, yy_train)
#
# print('###### optimal weights and the corresponding training error - ridge reg')
# print('Optimal weights - Ridge: \n', w_opt)
# print('Training error - Ridge: \n', err_train)

#########################3  Student Task. Lasso Regression.
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

m=20
n=10
X,y = GetFeaturesLabels(m,n)  # read in m data points using n features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) # 80% training and 20% test
alpha_val = 1
alpha_scaled = 2*n

print('###############   ''##########linear reg')
lr = LinearRegression()
print('lr.fit')
print(lr.fit(X_train, y_train))
print('lr.predict(X_train)')
print(lr.predict(X_train))
print('lr.predict(X_val')
print(lr.predict(X_val))
y_pred_tr = lr.predict(X_train)
y_pred_val = lr.predict(X_val)
print('lr msq train')
print(mean_squared_error(y_train, y_pred_tr))
print('lr msq validate')
print(mean_squared_error(y_val, y_pred_val))


print('###############################ridge reg')
rr = Ridge(alpha=alpha_scaled, fit_intercept=True)
rr.fit(X_train, y_train)
print('rr.predict(X_train)')
print(rr.predict(X_train))
print('rr.predict(X_val)')
print(rr.predict(X_val))
y_pred_tr = rr.predict(X_train)
y_pred_val = rr.predict(X_val)
print('rr msq train')
print(mean_squared_error(y_train, y_pred_tr))
print('rr msq validate')
print(mean_squared_error(y_val, y_pred_val))


print('###############################lasso reg')
print('###############################lasso reg')

lasso = Lasso(alpha=alpha_val*0.5, fit_intercept=False)
lasso.fit(X_train, y_train)
print('lasso.predict(X_train)')
print(lasso.predict(X_train))
print('lasso.predict(X_val)')
print(lasso.predict(X_val))
y_pred_tr = lasso.predict(X_train)
y_pred_val = lasso.predict(X_val)
print('lasso msq train')
print(mean_squared_error(y_train, y_pred_tr))
print('lasso msq validate')
print(mean_squared_error(y_val, y_pred_val))


w_opt = lasso.coef_
training_error = mean_squared_error(y_pred_tr, y_train)

print('###### optimal weights and the corresponding training error - lasso reg')
print('Optimal weights - Lasso: \n', w_opt)
print('Training error - Lasso: \n', training_error)
print('###############################lasso reg end')
print('###############################lasso reg end')
print('\n')
print('###############################  Student Task. Tuning Lasso Parameter.')

#n = 10
alpha_values = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 1, 3, 10, 10e3])
nr_values = len(alpha_values)
err_val = np.zeros([nr_values,1])
err_train = np.zeros([nr_values,1])
#w_opt = np.zeros([n,1])
### STUDENT TASK ###
# YOUR CODE HERE
#raise NotImplementedError()

print('##### Starting for loop ###')
for n in range (nr_values):
    print('loop %s begins with alpha %s' %(n,alpha_values[n]) )
    lasso = Lasso(alpha=alpha_values[n]*0.5, fit_intercept=False)
    lasso.fit(X_train, y_train)
    y_pred_tr = lasso.predict(X_train)
    y_pred_val = lasso.predict(X_val)
    print('lasso.predict(X_tr)')
    print(y_pred_tr)
    print('lasso.predict(X_val)')
    print(y_pred_val)
    err_train[n] = mean_squared_error(y_train, y_pred_tr)
    err_val[n] = mean_squared_error(y_val, y_pred_val)
    print('err_train')
    print(err_train)
    print('err_val')
    print(err_val)
    print('lasso.coef_')
    print(lasso.coef_)
    print('loop %s ends' % n)
    print('####################################')

print(np.argmin(err_val))
lasso = Lasso(alpha=alpha_values[np.argmin(err_val)]*0.5, fit_intercept=False).fit(X_train, y_train)
y_pred_validate = lasso.predict(X_val)
w_opt = lasso.coef_

print('lasso.coef_---')
print(lasso.coef_)

#Perform some sanity checks on the outputs
assert len(err_train) == 9, "'w_opts' has wrong shape"
assert len(err_val) == 9, "'w_opts' has wrong shape"
assert len(w_opt) == 10, "'w_opts' has wrong shape"

print('Sanity check tests passed!')
#

# Plot the training and validation errors
plt.plot(alpha_values, err_train, marker='o', label='training error')
plt.plot(alpha_values, err_val, marker='o', label='validation error')
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$E(\alpha)$')
plt.legend()
plt.show()