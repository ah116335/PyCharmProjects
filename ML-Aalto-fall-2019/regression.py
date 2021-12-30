from 	sklearn import linear_model
from 	sklearn.datasets import make_regression
import 	requests
import 	pandas as pd
from 	collections import OrderedDict
import 	matplotlib.pyplot as plt
from 	sklearn import datasets
import 	numpy as np
from 	matplotlib.colors import LogNorm
from 	sklearn.mixture import GaussianMixture


linnerud = datasets.load_linnerud()  # load Linnerud dataset into dictionary `linnerud`
X = linnerud['data']    # read out feature vectors stored under key 'data'
Y = linnerud['target']  # read out label values stored under key 'target'

x = Y.T[0] # weight (in Lbs) for each athlete
y = X.T[0] # number of chin ups for each athlete

x = x.reshape(-1,1)  # convert to numpy array of shape (m,1)
y = y.reshape(-1,1)  # convert to numpy array of shape (m,1)
x = x*0.453 # convert Lbs to Kg


# plot regression dataset

plt.rc('font', size=10)

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(x, y)
y_pred = reg.predict(x)

print("optimal weight w =", reg.coef_)


fig, axes = plt.subplots(1, 1, figsize=(8, 4))
axes.scatter(x, y, label='data points')
axes.plot(x, y_pred, color='green', label='optimal linear predictor')


# indicate error bars

axes.plot((x[0], x[0]), (y[0], y_pred[0]), color='red', label='errors') # add label to legend
for i in range(len(x)-1):
    lineXdata = (x[i+1], x[i+1]) # same X
    lineYdata = (y[i+1], y_pred[i+1]) # different Y
    axes.plot(lineXdata, lineYdata, color='red')

axes.legend()
axes.set_xlabel("feature x (body weight)")
axes.set_ylabel("label y (number of chin-ups)")
#plt.show()


# generate some synthetic dataset
syn_x, syn_y = make_regression(n_samples=3000, n_features=1, noise=30)
syn_y = syn_y + 10*np.ones(3000).reshape(syn_y.shape)
syn_x = syn_x + 10*np.ones(3000).reshape(syn_x.shape)

# plot regression datase

plt.rc('font', size=10)

reg = linear_model.LinearRegression(fit_intercept=False)
reg_intercept = linear_model.LinearRegression(fit_intercept=True)
reg = reg.fit(syn_x, syn_y)
reg_intercept = reg_intercept.fit(syn_x, syn_y)
x_grid = np.linspace(-1, 16, num=100).reshape(-1,1)
y_pred = reg.predict(x_grid)
y_pred_intercept = reg_intercept.predict(x_grid)

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
axes.scatter(syn_x, syn_y, label='data points')
axes.plot(x_grid, y_pred, color='green', label='no intercept')
axes.plot(x_grid, y_pred_intercept, color='red', label='with intercept')

axes.legend(loc='upper left')
axes.set_xlabel("feature x")
axes.set_ylabel("label y")
axes.axhline(y=0, color='k',linestyle=':')
axes.axvline(x=0, color='k',linestyle=':')
#plt.show()

n_samples = 300

np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = GaussianMixture(n_components=2, covariance_type='full', random_state=1)
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
