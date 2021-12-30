import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import warnings

import sys

# stdoutOrigin=sys.stdout
# sys.stdout = open("log.txt", "w")

#Initialize variables
warnings.filterwarnings("ignore")
polynomialdegree=1
polynomial_features = PolynomialFeatures(degree=polynomialdegree, include_bias=False)
testsize = 0.25
randomstate = random.randint(11, 999999999)
#randomstate = 5162
excelfile = 'bicycle.xlsx'
huber_epsilon=1

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

X_train, X_val, y_train, y_val = train_test_split(numpyExDataX, numpyExDataY, test_size=testsize, random_state=randomstate)

print('Calculating Linear regression')
LRregression = LinearRegression(fit_intercept=True)
LRregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", LRregression)])
LRregTr.fit(X_train, y_train)
LRmse_t    = mean_squared_error(y_train,  LRregTr.predict(X_train))
LRmae_t    = mean_absolute_error(y_train, LRregTr.predict(X_train))
LRmse_v   = mean_squared_error(y_val,  LRregTr.predict(X_val))
LRmae_v   = mean_absolute_error(y_val, LRregTr.predict(X_val))

print('Calculating Huberregression')
HUregression = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon, max_iter=250)
HUregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", HUregression)])
HUregTr.fit(X_train, y_train)
HUmse_t    = mean_squared_error(y_train,  HUregTr.predict(X_train))
HUmae_t    = mean_absolute_error(y_train, HUregTr.predict(X_train))
HUmse_v   = mean_squared_error(y_val,  HUregTr.predict(X_val))
HUmae_v   = mean_absolute_error(y_val, HUregTr.predict(X_val))

print('Calculating Ridgeregression')
RIregression = Ridge(alpha=1, fit_intercept=True)
RIregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", RIregression)])
RIregTr.fit(X_train, y_train)
RImse_t    = mean_squared_error(y_train,  RIregTr.predict(X_train))
RImae_t    = mean_absolute_error(y_train, RIregTr.predict(X_train))
RImse_v   = mean_squared_error(y_val,  RIregTr.predict(X_val))
RImae_v   = mean_absolute_error(y_val, RIregTr.predict(X_val))


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4), sharey=True, sharex=True, dpi=100, squeeze=True)

## Plotting LR
X_grid = np.linspace(12,26,100).reshape((-1,1))
ax1.plot(X_grid, LRregTr.predict(X_grid), label="Model")
ax1.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax1.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax1.set_xlabel('Temperature/centigrade')
ax1.set_ylabel('Laptime/minutes')
ax1.set_title('LinearRegression')
ax1.legend()

## Plotting HUr
ax2.plot(X_grid, HUregTr.predict(X_grid), label="Model")
ax2.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax2.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax2.set_xlabel('Temperature/centigrade')
ax2.set_ylabel('Laptime/minutes')
ax2.set_title('HuberRegression')
ax2.legend()

## Plotting RIr
ax3.plot(X_grid, RIregTr.predict(X_grid), label="Model")
ax3.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax3.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax3.set_xlabel('Temperature/centigrade')
ax3.set_ylabel('Laptime/minutes')
ax3.set_title('RidgeRegression')
ax3.legend()

plt.show()

print("Calculations complete")
print('Using randomstate', randomstate)
print('Amount of data points is' , len(numpyExDataX))
print('Testsize is', testsize)
print('Amount of training datapoints is', len(X_train))
print('Amount of validation datapoints is', len(X_val))
print('MSE Training_error for Linearregression is', LRmse_t)
print('MAE Training_error Linearregression is', LRmae_t)
print('MSE Validation_error Linearregression is', LRmse_v)
print('MAE Validation error Linearregression is', LRmae_v)

print("###  ###")
print('MSE Training_error for Huberregression is', HUmse_t)
print('MAE Training_error Huberregression is', HUmae_t)
print('MSE Validation_error Huberregression is', HUmse_v)
print('MAE Validation error Huberregression is', HUmae_v)

print("### :-) ###")
print('MSE Training_error for Ridgeregression is', RImse_t)
print('MAE Training_error Ridgeregression is', RImae_t)
print('MSE Validation_error Ridgeregression is', RImse_v)
print('MAE Validation error Ridgeregression is', RImae_v)

print("##########################################################")
print("Calculating scores")
print('Linearregression score is:',LRregTr.score(X_val, y_val))
print('Huberregression score is:', HUregTr.score(X_val, y_val))
print('Ridgeregression score is:', RIregTr.score(X_val, y_val))

print("##########################################################")
print("Calculating Predictions")
print('LR Laptime prediction for temp 20C is', LRregTr.predict(20))
LR_y_pred = LRregTr.predict(numpyExDataPred)
print(numpyExDataPred.T)
print(LR_y_pred.round(decimals=1).T)

print('HU Laptime prediction for temp 20C is', HUregTr.predict(20))
HU_y_pred = HUregTr.predict(numpyExDataPred)
print(numpyExDataPred.T)
print(HU_y_pred.round(decimals=1).T)

print('RI Laptime prediction for temp 20C is', RIregTr.predict(20))
RI_y_pred = RIregTr.predict(numpyExDataPred)
print(numpyExDataPred.T)
print(RI_y_pred.round(decimals=1).T)

# sys.stdout.close()
# sys.stdout=stdoutOrigin