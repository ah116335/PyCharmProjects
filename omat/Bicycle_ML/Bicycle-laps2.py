import random
import 	pandas as pd
import 	matplotlib.pyplot as plt
import 	numpy as np
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

#Initialize variables
polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
testsize = 0.3
#randomstate = random.randint(11, 999999999)
randomstate = 12234
print('Randomstate =', randomstate)
excelfile = 'bicycle.xlsx'
huber_epsilon=1
regressor = '1huber' # huber or linear


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

print('Data points ' , len(numpyExDataX))

X_train, X_val, y_train, y_val = train_test_split(numpyExDataX, numpyExDataY, test_size=testsize, random_state=randomstate)

if regressor is 'huber':
    print('Using Huberregression with epsilon', huber_epsilon)
    regression = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon, max_iter=250)
else :
    print("Using Linearregression")
    regression = LinearRegression(fit_intercept=True)

regTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", regression)])
print(regTr)

print("##########################################################")
print("Calculating Training Regression.fit")
regTr.fit(X_train, y_train)

mse_t    = mean_squared_error(y_train,  regTr.predict(X_train))
mae_t    = mean_absolute_error(y_train, regTr.predict(X_train))

print("##########################################################")
print("Calculating Validation")
mse_v   = mean_squared_error(y_val,  regTr.predict(X_val))
mae_v   = mean_absolute_error(y_val, regTr.predict(X_val))

## Plotting
X_grid = np.linspace(12,26,100).reshape((-1,1))
plt.plot(X_grid, regTr.predict(X_grid), label="Model")
plt.scatter(X_train, y_train, c='blue', s=10, label='Training data')
plt.scatter(X_val, y_val, c='red', s=10, label='Validation data')
plt.xlabel('Temperature/centigrade')
plt.ylabel('Laptime/minutes')
plt.legend()
plt.show()

print("Calculations complete")
print('Amount of training datapoints is', len(X_train))
print('Amount of validation datapoints is', len(X_val))
print('MSE Training_error ', mse_t)
print('MAE Training_error ', mae_t)
print('MSE Validation_error ', mse_v)
print('MAE Validation error ', mae_v)
print('Predict temp 20', regTr.predict(20))
print("### Done ###")

print("##########################################################")
print("Calculating Prediction")
y_pred = regTr.predict(numpyExDataPred)
print(numpyExDataPred.T)
print(y_pred.round(decimals=1).T)



