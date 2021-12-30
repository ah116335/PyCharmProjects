import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
dataset.head()

X = dataset.iloc[:,:-1].values  #independent variable array
y = dataset.iloc[:,1].values  #dependent variable vector

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train) #actually produces the linear eqn for the data
y_pred = regressor.predict(X_test)

print('y_pred', y_pred)
print('y_test', y_test)

# plot for the TRAIN
plt.scatter(X_train, y_train, color='red')  # plotting the observation line
plt.plot(X_train, regressor.predict(X_train), color='blue')  # plotting the regression line
plt.title("Salary vs Experience (Training set)")  # stating the title of the graph
plt.xlabel("Years of experience")  # adding the name of x-axis
plt.ylabel("Salaries")  # adding the name of y-axis
plt.show()  # specifies end of graph

# plot for the TEST
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # plotting the regression line
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Years of experience")
plt.ylabel("Salaries")
plt.show()

