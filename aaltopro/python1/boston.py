import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


polynomialdegree=1
polynomial_features = PolynomialFeatures(degree=polynomialdegree, include_bias=False)


#read csv into pandas dataframe
df_inputdata = pd.read_csv('bostondataset.csv')
df_columns = (df_inputdata.head(0))
m=np.shape(df_inputdata)[0]
n=np.shape(df_inputdata)[1]
#print(type(df_columns))

#convert last column (price) to numpy array
np_price_column = df_inputdata.iloc[: , -1].values
#print(price_column)

#convert 13 first columns (the features) to numpy array
N=13
np_features_columns = df_inputdata.iloc[: , :N].values

print("")
print("Sample size m = ", m)
print("Feature length n = ", n)
print('Polynomial degree is', polynomialdegree)

X_train, X_val, y_train, y_val = train_test_split(np_features_columns, np_price_column, test_size = 0.7,
                                                  random_state=6783)

reg = LinearRegression(fit_intercept=True)

LRregTr = Pipeline([("polynomial_features", polynomial_features),("linear_regression", reg)])
LRregTr.fit(X_train, y_train)

LRmse_t    = mean_squared_error(y_train,  LRregTr.predict(X_train))
print("reg.score(X_train, y_train) is" , LRregTr.score(X_train, y_train))
print("LRmse_t is" , LRmse_t)

LRmse_v = mean_squared_error(y_val, LRregTr.predict(X_val))
#print("reg.score(X_train, y_train) is" , LRregTr.score(X_train, y_train))
print("LRmse_v is" , LRmse_v)

# X_grid = 1,1
# ## Plotting LR
# plt.plot(X_grid, LRregTr.predict(X_grid), label="Model")
# plt.scatter(X_train, y_train, c='blue', s=10, label='Training data')
# #ax1.scatter(X_val, y_val, c='red', s=10, label='Validation data')
# #ax1.scatter(X_test, y_test, c='yellow', s=10, label='Test data')
# plt.set_xlabel('Temperature/centigrade')
# plt.set_ylabel('Laptime/minutes')
# plt.set_title('LinearRegression')
# plt.set_facecolor(ax_face_color)
# plt.legend()
