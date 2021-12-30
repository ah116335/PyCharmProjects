import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, BayesianRidge, RANSACRegressor, Lars
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import warnings


import sys

# stdoutOrigin=sys.stdout
# sys.stdout = open("log.txt", "w")

#Initialize variables
warnings.filterwarnings("ignore")
polynomialdegree=4
polynomial_features = PolynomialFeatures(degree=polynomialdegree, include_bias=False)
testsize = 0.1
validationsize = 0.2
#randomstate = random.randint(11, 999999999)
randomstate = 10186
excelfile = 'bicycle.xlsx'
huber_epsilon=1
ridgeregularizationalpha=22
huberregularizationalpha=1
ax_face_color='lightgrey'
clusters = 3
rig = 0 ## switch between 1 and 0
data_colors = ['orangered','dodgerblue','springgreen'] # colors for data points

print('Using excelfile', excelfile)

def loadexcel(filename, sheetname=0):
    sheetname='Bicycle'
    #sheetname='BicycleP'
    df1 = pd.read_excel(filename, sheet_name=sheetname, usecols="J",convert_float=True)
    df2 = pd.read_excel(filename, sheet_name=sheetname, usecols="K",convert_float=True)
    df3 = pd.read_excel(filename, sheet_name='Predict', usecols='A',convert_float=True)
    exceldataX = df1.values
    exceldataY = df2.values
    exceldataPred = df3.values
    return exceldataX, exceldataY, exceldataPred

ExDataX, ExDataY, ExDataPred = loadexcel(excelfile)

numpyExDataX = np.array(ExDataX).reshape((-1, 1))
numpyExDataY = np.array(ExDataY)
numpyExDataPred = np.array(ExDataPred).reshape((-1, 1))

## First split data set into training and intermediate_validation sets
X_train, X_val_i, y_train, y_val_i = train_test_split(numpyExDataX, numpyExDataY, test_size=(validationsize+testsize), random_state=randomstate)

## Then split the intermediate validation set into actual validation and test sets in 1:2 ratio)
X_val, X_test, y_val, y_test = train_test_split(X_val_i, y_val_i, test_size=(validationsize+testsize), random_state=randomstate)

numpy_X_train_y_train = np.c_[X_train, y_train]

print('Calculating Linearregression')
LRregression = LinearRegression(fit_intercept=True)
LRregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", LRregression)])
LRregTr.fit(X_train, y_train)
LRmse_t    = mean_squared_error(y_train,  LRregTr.predict(X_train))
LRmae_t    = mean_absolute_error(y_train, LRregTr.predict(X_train))
LRmse_v   = mean_squared_error(y_val,  LRregTr.predict(X_val))
LRmae_v   = mean_absolute_error(y_val, LRregTr.predict(X_val))
LRmse_te   = mean_squared_error(y_test,  LRregTr.predict(X_test))
LRmae_te   = mean_absolute_error(y_test, LRregTr.predict(X_test))

print('Calculating Huberregression')
HUregression = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon, max_iter=2500, alpha=huberregularizationalpha)
HUregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", HUregression)])
HUregTr.fit(X_train, y_train)
HUmse_t    = mean_squared_error(y_train,  HUregTr.predict(X_train))
HUmae_t    = mean_absolute_error(y_train, HUregTr.predict(X_train))
HUmse_v   = mean_squared_error(y_val,  HUregTr.predict(X_val))
HUmae_v   = mean_absolute_error(y_val, HUregTr.predict(X_val))
HUmse_te   = mean_squared_error(y_test,  HUregTr.predict(X_test))
HUmae_te   = mean_absolute_error(y_test, HUregTr.predict(X_test))

print('Calculating Ridgeregression')
RIregression = Ridge(alpha=ridgeregularizationalpha, fit_intercept=True)
RIregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", RIregression)])
RIregTr.fit(X_train, y_train)
RImse_t    = mean_squared_error(y_train,  RIregTr.predict(X_train))
RImae_t    = mean_absolute_error(y_train, RIregTr.predict(X_train))
RImse_v   = mean_squared_error(y_val,  RIregTr.predict(X_val))
RImae_v   = mean_absolute_error(y_val, RIregTr.predict(X_val))
RImse_te   = mean_squared_error(y_test,  RIregTr.predict(X_test))
RImae_te   = mean_absolute_error(y_test, RIregTr.predict(X_test))

print('Calculating Lassoregression')
LAregression = Lasso(alpha=ridgeregularizationalpha, fit_intercept=True)
LAregTr = Pipeline([("polynomial_features", polynomial_features),("lasso_regression", LAregression)])
LAregTr.fit(X_train, y_train)
LAmse_t    = mean_squared_error(y_train,  LAregTr.predict(X_train))
LAmae_t    = mean_absolute_error(y_train, LAregTr.predict(X_train))
LAmse_v   = mean_squared_error(y_val,  LAregTr.predict(X_val))
LAmae_v   = mean_absolute_error(y_val, LAregTr.predict(X_val))
LAmse_te   = mean_squared_error(y_test,  LAregTr.predict(X_test))
LAmae_te   = mean_absolute_error(y_test, LAregTr.predict(X_test))

#Generating plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,4), sharey=False, sharex=True, dpi=100, squeeze=True)
X_grid = np.linspace(12,26,100).reshape((-1,1))

## Plotting LR
ax1.plot(X_grid, LRregTr.predict(X_grid), label="Model")
ax1.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax1.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax1.scatter(X_test, y_test, c='yellow', s=10, label='Test data')
ax1.set_xlabel('Temperature/centigrade')
ax1.set_ylabel('Laptime/minutes')
ax1.set_title('LinearRegression')
ax1.set_facecolor(ax_face_color)
ax1.legend()

## Plotting HUr
ax2.plot(X_grid, HUregTr.predict(X_grid), label="Model")
ax2.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax2.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax2.scatter(X_test, y_test, c='yellow', s=10, label='Test data')
ax2.set_xlabel('Temperature/centigrade')
ax2.set_ylabel('Laptime/minutes')
ax2.set_title('HuberRegression')
ax2.set_facecolor(ax_face_color)
ax2.legend()

## Plotting RIr
ax3.plot(X_grid, RIregTr.predict(X_grid), label="Model")
ax3.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax3.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax3.scatter(X_test, y_test, c='yellow', s=10, label='Test data')
ax3.set_xlabel('Temperature/centigrade')
ax3.set_ylabel('Laptime/minutes')
ax3.set_title('RidgeRegression')
ax3.set_facecolor(ax_face_color)
ax3.legend()

## Plotting LAr
ax4.plot(X_grid, LAregTr.predict(X_grid), label="Model")
ax4.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax4.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax4.scatter(X_test, y_test, c='yellow', s=10, label='Test data')
ax4.set_xlabel('Temperature/centigrade')
ax4.set_ylabel('Laptime/minutes')
ax4.set_title('LassoRegression')
ax4.set_facecolor(ax_face_color)
ax4.legend()



print("*** Calculations complete")
print(" ")
print("Numerical values used are")
print(' Randomstate is', randomstate)
print(' Polynomial degree is', polynomialdegree)
print(' Amount of data points is' , len(numpyExDataX))
print(' Amount of training points is' , len(X_train))
print(' Amount of validation points is' , len(X_val))
print(' Amount of test points is' , len(X_test))
print(' Validationsize is', validationsize, 'and testsize is', testsize)

print(" ")
print('MSE Training_error for Linearregression is', LRmse_t)
print('MAE Training_error for Linearregression is', LRmae_t)
print('MSE Validation_error for Linearregression is', LRmse_v)
print('MAE Validation error for Linearregression is', LRmae_v)
print('MSE Test_error for Linearregression is', LRmse_te)
print('MAE Test_error for Linearregression is', LRmae_te)

print("###  ###")
print('MSE Training_error for Huberregression is', HUmse_t)
print('MAE Training_error for Huberregression is', HUmae_t)
print('MSE Validation_error for Huberregression is', HUmse_v)
print('MAE Validation error for Huberregression is', HUmae_v)
print('MSE Test_error for Huberregression is', HUmse_te)
print('MAE Test_error for Huberregression is', HUmae_te)

print("######")
print('MSE Training_error for Ridgeregression is', RImse_t)
print('MAE Training_error for Ridgeregression is', RImae_t)
print('MSE Validation_error for Ridgeregression is', RImse_v)
print('MAE Validation error for Ridgeregression is', RImae_v)
print('MSE Test_error for Ridgeregression is', RImse_te)
print('MAE Test_error for Ridgeregression is', RImae_te)

print("### :-) ###")
print('MSE Training_error for Lassoregression is', LAmse_t)
print('MAE Training_error for Lassoregression is', LAmae_t)
print('MSE Validation_error for Lassoregression is', LAmse_v)
print('MAE Validation error for Lassoregression is', LAmae_v)
print('MSE Test_error for Lassoregression is', LAmse_te)
print('MAE Test_error for Lassoregression is', LAmae_te)


 
print("##########################################################")
print("Calculating scores")
print('Linearregression score for training dataset is:',LRregTr.score(X_train, y_train))
print('Huberregression score for training dataset is:', HUregTr.score(X_train, y_train))
print('Ridgeregression score for training dataset is:', RIregTr.score(X_train, y_train))
print('LAssoregression score for training dataset is:', LAregTr.score(X_train, y_train))
print("##########################################################")
print('Linearregression score for validation dataset is:',LRregTr.score(X_val, y_val))
print('Huberregression score for validation dataset is:', HUregTr.score(X_val, y_val))
print('Ridgeregression score for validation dataset is:', RIregTr.score(X_val, y_val))
print('LAssoregression score for validation dataset is:', LAregTr.score(X_val, y_val))
print("##########################################################")
print('Linearregression score for test dataset is:',LRregTr.score(X_test, y_test))
print('Huberregression score for test dataset is:', HUregTr.score(X_test, y_test))
print('Ridgeregression score for test dataset is:', RIregTr.score(X_test, y_test))
print('LAssoregression score for test dataset is:', LAregTr.score(X_test, y_test))

print("##########################################################")

# print("Calculating Predictions")
# print('LR Laptime prediction for temp 20C is', LRregTr.predict(20))
# LR_y_pred = LRregTr.predict(numpyExDataPred)
# print(numpyExDataPred.T)
# print(LR_y_pred.round(decimals=1).T)
#
# print('HU Laptime prediction for temp 20C is', HUregTr.predict(20))
# HU_y_pred = HUregTr.predict(numpyExDataPred)
# print(numpyExDataPred.T)
# print(HU_y_pred.round(decimals=1).T)
#
# print('RI Laptime prediction for temp 20C is', RIregTr.predict(20))
# RI_y_pred = RIregTr.predict(numpyExDataPred)
# print(numpyExDataPred.T)
# print(RI_y_pred.round(decimals=1).T)

print("END ######################################################")
print("##########################################################")
print()
# sys.stdout.close()
# sys.stdout=stdoutOrigin
plt.show()