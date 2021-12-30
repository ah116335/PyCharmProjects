import random
import 	pandas as pd
import 	matplotlib.pyplot as plt
import 	numpy as np
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


polynomial_features = PolynomialFeatures(degree=2, include_bias=False)

testsize = 0.2
#randomstate = random.randint(11, 999999999)
randomstate = 123456
excelfile = 'bicycle.xlsx'
print('Randomstate =', randomstate)
huber_epsilon=1
regressor = 'huber1' #if anything else than 'huber', -> linearregression

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
    regTr = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon)
    regVal = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon)
else :
    print("Using Linearregression")
    regTr = LinearRegression(fit_intercept=True)
    regVal = LinearRegression(fit_intercept=True)


print("##########################################################")
print("Calculating Training Regression.fit")
regTr = regTr.fit(X_train, y_train)
training_errorTr = mean_squared_error(y_train, regTr.predict(X_train))
print('Regression_coef_', regTr.coef_)
print('Regression_intercept_', regTr.intercept_)
print('Training_error ', training_errorTr)
print('Predict temp 20', regTr.predict(20))

print("##########################################################")
print("Calculating Validation Regression.fit")
regVal = regVal.fit(X_val, y_val)
training_errorVal = mean_squared_error(y_val, regVal.predict(X_val))
print('Regression_coef_', regVal.coef_)
print('Regression_intercept_', regVal.intercept_)
print('Validation_error ', training_errorVal)
print('Predict temp 20', regVal.predict(20))

print("##########################################################")
print("Calculating Prediction")
y_pred = regTr.predict(numpyExDataPred)
print(numpyExDataPred.T)
print(y_pred.round(decimals=1).T)

print("Calculations complete")
print('Amount of training datapoints is', len(X_train))
print('Amount of validation datapoints is', len(X_val))
print("### Done ###")

plt.grid()
plt.scatter(X_train, y_train, color='blue',label='Training data')
plt.scatter(X_val, y_val, color='red',label='Validation data')

plt.title("Bicycle laps")
plt.xlabel("Temperature (Celsius)")  # adding the name of x-axis
plt.ylabel("Laptime (minutes)")  # adding the name of y-axis
plt.legend()

plt.plot(X_train, regTr.predict(X_train), color='blue')  # plotting the regression line
plt.plot(X_val, regVal.predict(X_val), color='red')  # plotting the regression line
plt.show()


# plot for the TRAIN
# plt.scatter(X_train, y_train, color='red')  # plotting the observation line
# plt.plot(X_train, regTr.predict(X_train), color='blue')  # plotting the regression line
# plt.title("Training set")  # stating the title of the graph
# plt.xlabel("Temp")  # adding the name of x-axis
# plt.ylabel("Laptime")  # adding the name of y-axis
# plt.show()  # specifies end of graph
# def ScatterPlots():
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#     axes[0].scatter(X_train, y_train, label='Training data', c='b')
#     axes[0].legend()
#     axes[0].scatter(X_val, y_val, label='Validation data', c='r')
#     axes[0].legend()
#     return axes
#
# axes = ScatterPlots()