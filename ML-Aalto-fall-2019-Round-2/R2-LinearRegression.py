from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math

reg = LinearRegression(fit_intercept=False)
reg = reg.fit(X, y)
training_error = mean_squared_error(y, reg.predict(X))

display(Math(r'$\mathbf{w}_{\rm opt} ='))
optimal_weight = reg.coef_
optimal_weight = optimal_weight.reshape(-1,1)
print(optimal_weight)
print("\nThe resuling training error is ",training_error)