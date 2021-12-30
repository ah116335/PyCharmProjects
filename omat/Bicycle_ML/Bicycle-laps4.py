import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, BayesianRidge, RANSACRegressor, Lars
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
testsize = 0.2
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

plt.scatter(numpyExDataX, numpyExDataY)
plt.xlabel('Temperature/centigrade')
plt.ylabel('Laptime/minutes')
#plt.set_title('LassoRegression')
plt.legend()
plt.show()

print('Calculating Linearregression')
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

print('Calculating Lassoregression')
LAregression = Lasso(alpha=1, fit_intercept=True)
LAregTr = Pipeline([("polynomial_features", polynomial_features),("lasso_regression", LAregression)])
LAregTr.fit(X_train, y_train)
LAmse_t    = mean_squared_error(y_train,  LAregTr.predict(X_train))
LAmae_t    = mean_absolute_error(y_train, LAregTr.predict(X_train))
LAmse_v   = mean_squared_error(y_val,  LAregTr.predict(X_val))
LAmae_v   = mean_absolute_error(y_val, LAregTr.predict(X_val))
# print('Printing Lassoregression coeff')
# print(LAregression.coef_)
# print(LAregression.intercept_)

print('Calculating Ransacregression')
RAregression = RANSACRegressor()
RAregTr = Pipeline([("polynomial_features", polynomial_features),("RANSAC_regression", RAregression)])
RAregTr.fit(X_train, y_train)
RAmse_t    = mean_squared_error(y_train,  RAregTr.predict(X_train))
RAmae_t    = mean_absolute_error(y_train, RAregTr.predict(X_train))
RAmse_v   = mean_squared_error(y_val,  RAregTr.predict(X_val))
RAmae_v   = mean_absolute_error(y_val, RAregTr.predict(X_val))

print('Calculating BayesianRidge')
BAregression = BayesianRidge()
BAregTr = Pipeline([("polynomial_features", polynomial_features),("BayesianRidge_regression", BAregression)])
BAregTr.fit(X_train, y_train)
BAmse_t    = mean_squared_error(y_train,  BAregTr.predict(X_train))
BAmae_t    = mean_absolute_error(y_train, BAregTr.predict(X_train))
BAmse_v   = mean_squared_error(y_val,  BAregTr.predict(X_val))
BAmae_v   = mean_absolute_error(y_val, BAregTr.predict(X_val))

print('Calculating LARS')
LARSregression = Lars()
LARSregTr = Pipeline([("polynomial_features", polynomial_features),("LassoLARS regression", LARSregression)])
LARSregTr.fit(X_train, y_train)
LARSmse_t    = mean_squared_error(y_train,  LARSregTr.predict(X_train))
LARSmae_t    = mean_absolute_error(y_train, LARSregTr.predict(X_train))
LARSmse_v   = mean_squared_error(y_val,  LARSregTr.predict(X_val))
LARSmae_v   = mean_absolute_error(y_val, LARSregTr.predict(X_val))


fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,7, figsize=(10,4), sharey=True, sharex=True, dpi=100, squeeze=True)
X_grid = np.linspace(12,26,100).reshape((-1,1))

## Plotting LR
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

## Plotting LAr
ax4.plot(X_grid, LAregTr.predict(X_grid), label="Model")
ax4.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax4.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax4.set_xlabel('Temperature/centigrade')
ax4.set_ylabel('Laptime/minutes')
ax4.set_title('LassoRegression')
ax4.legend()

## Plotting RAr
ax5.plot(X_grid, RAregTr.predict(X_grid), label="Model")
ax5.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax5.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax5.set_xlabel('Temperature/centigrade')
ax5.set_ylabel('Laptime/minutes')
ax5.set_title('RANSACRegression')
ax5.legend()

## Plotting BAr
ax6.plot(X_grid, BAregTr.predict(X_grid), label="Model")
ax6.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax6.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax6.set_xlabel('Temperature/centigrade')
ax6.set_ylabel('Laptime/minutes')
ax6.set_title('BayesianRidge')
ax6.legend()

## Plotting LARS
ax7.plot(X_grid, LARSregTr.predict(X_grid), label="Model")
ax7.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax7.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax7.set_xlabel('Temperature/centigrade')
ax7.set_ylabel('Laptime/minutes')
ax7.set_title('LARS')
ax7.legend()



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

print("######")
print('MSE Training_error for Ridgeregression is', RImse_t)
print('MAE Training_error Ridgeregression is', RImae_t)
print('MSE Validation_error Ridgeregression is', RImse_v)
print('MAE Validation error Ridgeregression is', RImae_v)

print("### :-) ###")
print('MSE Training_error for Lassoregression is', LAmse_t)
print('MAE Training_error Lassoregression is', LAmae_t)
print('MSE Validation_error Lassoregression is', LAmse_v)
print('MAE Validation error Lassoregression is', LAmae_v)

print("### :-) ###")
print('MSE Training_error for RANSACregression is', RAmse_t)
print('MAE Training_error RANSACregression is', RAmae_t)
print('MSE Validation_error RANSACregression is', RAmse_v)
print('MAE Validation error RANSACregression is', RAmae_v)

print("### :-) ###")
print('MSE Training_error for BayesianRidge is', BAmse_t)
print('MAE Training_error BayesianRidge is', BAmae_t)
print('MSE Validation_error BayesianRidge is', BAmse_v)
print('MAE Validation error BayesianRidge is', BAmae_v)

print("### :-) ###")
print('MSE Training_error for LassoLARS is', LARSmse_t)
print('MAE Training_error LassoLARS is', LARSmae_t)
print('MSE Validation_error LassoLARS is', LARSmse_v)
print('MAE Validation error LassoLARS is', LARSmae_v)

print("##########################################################")
print("Calculating scores")
print('Linearregression score is:',LRregTr.score(X_val, y_val))
print('Huberregression score is:', HUregTr.score(X_val, y_val))
print('Ridgeregression score is:', RIregTr.score(X_val, y_val))
print('LAssoregression score is:', LAregTr.score(X_val, y_val))
print('RANSACregression score is:', RAregTr.score(X_val, y_val))
print('BayesianRidge score is:', BAregTr.score(X_val, y_val))
print('LassoLARS score is:', LARSregTr.score(X_val, y_val))

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