import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, BayesianRidge, RANSACRegressor, Lars
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import warnings
import time

filenamecsv = 'SARS-CoV-2_variants_eur.csv'
filenameexcel = 'SARS-CoV-2_variants_eur.xlsx'
sheetname = "Sheet7"
columns = 'A-J'
testsize = 0.3
validationsize = 0.3
#randomstate = random.randint(11, 999999999)
randomstate = 1283
polynomialdegree=1
polynomial_features = PolynomialFeatures(degree=polynomialdegree, include_bias=False)
ax_face_color='lightgrey'
huber_epsilon=1
ridgeregularizationalpha=2
huberregularizationalpha=1

def LoadMyExcel(filename, sheet=0):

    exceldata = None
#    df1 = pd.read_excel(filename, sheet_name=sheetname, usecols=columns)
    df1 = pd.read_excel(filename, sheet_name=sheetname)
    exceldata = df1.values
    return exceldata, df1

ExcelData, ExcelDFr = LoadMyExcel(filenameexcel)
numpyExDataX = np.array(ExcelData[:,4]).reshape((-1, 1))
numpyExDataY = np.array(ExcelData[:,5])

# start1 = time.time()
# df8 = pd.read_excel(excelfile, sheet_name='All')
# end1 = time.time()
# print('Excel:', end1 - start1)
#
# start2 = time.time()
# df9 = pd.read_csv(filenamecsv)
# end2 = time.time()
# print('CSV:', end2 - start2)


# print(ExData[10:12,0])        # print rows 10 and 11 , and column 0 only.  Note - indexing starts from 0
# ExDFr.to_csv('mymymy.csv')    # print a csv of dataframe ExDFr
# np.savetxt('ggg.csv', ExData, delimiter=",", fmt='%s')    #print a csv of numpy array ExData


X_train, X_val, y_train, y_val = train_test_split(numpyExDataX, numpyExDataY, test_size=(testsize), random_state=randomstate)


print('Calculating Linearregression')
LRregression = LinearRegression(fit_intercept=True)
LRregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", LRregression)])
LRregTr.fit(X_train, y_train)
LRmse_t    = mean_squared_error(y_train,  LRregTr.predict(X_train))
LRmse_v    = mean_squared_error(y_val,    LRregTr.predict(X_val))

sample = LRregTr.predict(10000)
print("")
print('Sample =', sample)



print('Calculating Huberregression')
HUregression = HuberRegressor(fit_intercept=True, epsilon=huber_epsilon, max_iter=2500, alpha=huberregularizationalpha)
HUregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", HUregression)])
HUregTr.fit(X_train, y_train)
HUmse_t    = mean_squared_error(y_train,  HUregTr.predict(X_train))
HUmse_v    = mean_squared_error(y_val,    HUregTr.predict(X_val))


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharey=True, sharex=True, dpi=100, squeeze=True)
X_grid = np.linspace(0,numpyExDataX.max(),100).reshape((-1,1))

## Plotting LR
ax1.plot(X_grid, LRregTr.predict(X_grid), label="Model")
ax1.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax1.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Days')
ax1.set_title('LinearRegression')
ax1.set_facecolor(ax_face_color)
ax1.legend()

## Plotting HUr
ax2.plot(X_grid, HUregTr.predict(X_grid), label="Model")
ax2.scatter(X_train, y_train, c='blue', s=10, label='Training data')
ax2.scatter(X_val, y_val, c='red', s=10, label='Validation data')
ax2.set_xlabel('Distance')
ax2.set_ylabel('Days')
ax2.set_title('HuberRegression')
ax2.set_facecolor(ax_face_color)
ax2.legend()

print("*** Calculations complete")
print(" ")
print("Numerical values used are")
print(' Randomstate', randomstate)
print(' Amount of data points is' , len(numpyExDataX))
print(' Amount of training points is' , len(X_train))
print(' Amount of validation points is' , len(X_val))
#print(' Amount of test points is' , len(X_test))
print(' Validationsize is', validationsize)

print(" ")
print('MSE Training_error for Linearregression is', LRmse_t)
print('MSE Validation_error for Linearregression is', LRmse_v)

print("###  ###")
print('MSE Training_error for Huberregression is', HUmse_t)
print('MSE Validation_error for Huberregression is', HUmse_v)


plt.show()