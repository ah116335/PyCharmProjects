# import required libraries (packages) for this exercise
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

m = 30  # number of images
dataset = np.zeros((m, 50, 50), dtype=np.uint8)  # create numpy array for images and fill with zeros
D = 50 * 50  # length of raw feature vectors
#print(dataset)

for i in range(1, m + 1):
    #print('starting round %d' %i)
    # with convert('L') we convert the images to grayscale
    try:
        img = Image.open('fruits/%s.jpg' % (str(i))).convert('L')  # read in image from jpg file
    except:
        img = Image.open('../../data/fruits/%s.jpg' % (str(i))).convert('L')  # read if you are doing exercise locally
    dataset[i - 1] = np.array(img, dtype=np.uint8)  # convert image to numpy array with greyscale values

#print('shape of dataset is ', dataset.shape)

#a=np.zeros((2,2))
#print(a)

Z = np.reshape(dataset, (m, -1))  # reshape the 50 x 50 pixels into a long numpy array of shape (2500,1)
# print("The shape of the datamatrix Z is", Z.shape)
# print(Z)

# display first three apple images (fruits1.jpg,fruits2.jpg,fruits3.jpg)
# and first three banana images (fruits16.jpg,fruits17.jpg,fruits18.jpg)

fig, ax = plt.subplots(3, 2, figsize=(10, 10), gridspec_kw={'wspace': 0, 'hspace': 0})
for i in range(3):
    for j in range(2):
        bitmap = Z[i + (15 * j), :]
        bitmap = np.reshape(bitmap, (50, 50))
        #ax[i,j].imshow(dataset[i+(15*j)], cmap='gray')
        ax[i, j].imshow(bitmap, cmap='gray')
        ax[i, j].axis('off')
#plt.show()

############################# STD TASK Compute PCA

print('############################# Starting student task Compute PCA ')

from sklearn.decomposition import PCA
import numpy as np

n = 10     #nr of principal components
m = 30
# YOUR CODE HERE
#raise NotImplementedError()

Z_hat = np.zeros((n,2500))
err_pca = np.zeros((m,1))
W_pca = None

# print('Z is', Z)
# print('Shape of Z is', Z.shape)

pca = PCA(n_components=n)
pca.fit(Z)
pca.transform(Z)
W_pca_OptimalCompressionMatrix = pca.components_

print('Optimal compression matrix W_pca is ' , W_pca_OptimalCompressionMatrix)
print('W_pca.shape' , W_pca_OptimalCompressionMatrix.shape)
print('W_pca transposed is', W_pca_OptimalCompressionMatrix.T)
print('W_pca.T.shape is ' , W_pca_OptimalCompressionMatrix.T.shape)

#####################################################Student Task. Reconstruction Error vs. Number of PC.
print('Student Task. Reconstruction Error vs. Number of PC')

#print('W_pca*Z is' , W_pca*Z)
# (Z_hat = W_pca.T * W_pca*Z)

for n_minus_1 in range(m):
    #print('starting round', n_minus_1+1)
    PCA_matrix_i = PCA(n_components = n_minus_1+1)
    PCA_matrix_i.fit(Z)
    PCA_W_pca_i_OptimalCompressionMatrix = PCA_matrix_i.components_
    PCA_Compressed_Z = PCA_matrix_i.transform(Z)
    Z_hat = np.dot(Z, PCA_W_pca_i_OptimalCompressionMatrix.T)
    PCA_matrix_i_reconstructed = PCA_matrix_i.inverse_transform(PCA_Compressed_Z)
    W_pca = PCA_W_pca_i_OptimalCompressionMatrix
    #print('W_pca.shape is', W_pca.shape)
    #print('Z_hat.shape is', Z_hat.shape)
    loss = None
    loss = ((Z - PCA_matrix_i_reconstructed)**2).sum()/m
    #print('loss is', loss)
    err_pca[n_minus_1] = loss

#print('err_pca.shape is', err_pca.shape)


assert err_pca.shape == (m, 1), "shape of err_pca is wrong."
assert err_pca[0] > err_pca[m-1], "values of err_pca are incorrect"
assert err_pca[0] > err_pca[1], "values of err_pca are incorrect"



#print('Sanity checks passed!')

import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")


## Input:
#  Z: Dataset
#  n: number of dimensions
#  m_pics: number of pics per class (Apple, Banana). Min 1, Max 15
def plot_reconstruct(Z, n, m_pics=3):
    # x=w*z
    X_pca = np.matmul(W_pca[:n, :], Z[:, :, None])
    # x_reversed=r*x
    X_reversed = np.matmul(W_pca[:n, :].T, X_pca)[:, :, 0]

    # Setup Figure size that scales with number of images
    fig = plt.figure(figsize=(10, 10))

    # Setup a (n_pics,2) grid of plots (images)
    gs = gridspec.GridSpec(1, 2 * m_pics)
    gs.update(wspace=0.0, hspace=0.0)
    for i in range(m_pics):
        for j in range(0, 2):
            # Add a new subplot
            ax = plt.subplot(gs[0, i + j * m_pics])
            # Insert image data into the subplot
            ax.imshow(np.reshape(X_reversed[i + (15 * j)], (50, 50)), cmap='gray', interpolation='nearest')
            # Remove x- and y-axis from each plot
            ax.set_axis_off()

    plt.subplot(gs[0, 0]).set_title("Reconstructed images using %d PCs:" % (n), size='large', y=1.08)
    # Render the plot
    #plt.show()


pca = PCA(n_components=m)  # create the object
pca.fit(Z)  # compute optimal transform W_PCA
W_pca = pca.components_

# The values of PCS n to plot for. You can change these to experiment
num_com = [1, 5, 50]
for n in num_com:
    # If you want to print different amount of pictures, you can change the value of m_pics. (1-15)
    #print(n)
    plot_reconstruct(Z, n, m_pics=3)

############################33Student Task. PCA with  𝑛=2 .
print('########################################## ')
print('Begin - Student Task. PCA with  𝑛=2 ')
print('########################################## ')

X_visual = np.zeros((m,2))
#print(Z)

PCA_matrix_n = PCA(n_components = 2)
PCA_matrix_n.fit(Z)
PCA_W_pca_n_OptimalCompressionMatrix = PCA_matrix_n.components_
PCA_Compressed_Z = PCA_matrix_n.transform(Z)
#Z_hat = np.matmul(PCA_matrix_i.transform(Z)[:,:i+1], PCA_matrix_i.components_[:i+1,:])
#Z_hat = np.dot(PCA_W_pca_i_OptimalCompressionMatrix[:i+1,:].T, PCA_matrix_i.components_)[:,:,0]
#Z_hat = np.dot(Z, PCA_W_pca_i_OptimalCompressionMatrix.T)
PCA_matrix_n_reconstructed = PCA_matrix_n.inverse_transform(PCA_Compressed_Z)
# print('Z is', Z)
# print('OptimalCompressionMatrix_ is ', PCA_W_pca_i_OptimalCompressionMatrix)
# print('Compressed_Z is', PCA_Compressed_Z)
# #print('Z_hat is', Z_hat)
# print('PCA_matrix_i_reconstructed is',PCA_matrix_i_reconstructed)
# print('shape of Z', Z.shape)
# print('shape of Z_compressed', PCA_Compressed_Z.shape)
# print('shape of Z_hat', Z_hat.shape)
#W_pca = PCA_W_pca_i_OptimalCompressionMatrix
#err_pca[i] =
loss = None
loss = ((Z - PCA_matrix_n_reconstructed)**2).sum()/m
print('loss is', loss)
err_pca[i] = loss
# total_loss = LA.norm((Z - PCA_matrix_i_reconstructed), None)
# print('total loss is', total_loss)
X_visual = PCA_Compressed_Z
# print('X_visual is', X_visual)
# print('shape of X_Visual is', X_visual.shape)
### STUDENT TASK ###
# YOUR CODE HERE
#raise NotImplementedError()

plt.gcf().clear()
plt.close('all')
##############################################Student Task. PCA with  𝑛=2 .
print('##########################################################')
print(' Starting Student Task. PCA with  𝑛=2 .')
print('##########################################################')

X_visual = np.zeros((m,2))

### STUDENT TASK ###
# YOUR CODE HERE
#raise NotImplementedError()
PCA_matrix_n = PCA(n_components = 2)
PCA_matrix_n.fit(Z)
PCA_W_pca_n_OptimalCompressionMatrix = PCA_matrix_n.components_
PCA_Compressed_Z = PCA_matrix_n.transform(Z)
#PCA_matrix_n_reconstructed = PCA_matrix_n.inverse_transform(PCA_Compressed_Z)
X_visual = PCA_Compressed_Z

#print('X_visual is', X_visual)
print('shape of X_Visual is', X_visual.shape)
print('##########################################################')


pca = PCA(n_components=m)
pca.fit(Z)
W_pca = pca.components_
X = np.dot(Z, W_pca.T)

X_PC12 = X[:, [0, 1]]
#X_PC12 = X[:]
#print(X_PC12)
X_PC89 = X[:, [7, 8]]
#print(X_PC89)

# plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
#
# plt.figure()
# plt.title('using first two PCs $x_{1}$ and $x_{2}$ as features')
# plt.scatter(X_PC12[:15, 0], X_PC12[:15, 1], c='r', marker='o', label='Apple')
# plt.scatter(X_PC12[15:, 0], X_PC12[15:, 1], c='y', marker='^', label='Banana')
# plt.legend()
# plt.xlabel('$x_{1}$')
# plt.ylabel('$x_{2}$')
# #plt.show()
#
# plt.figure()
# plt.title('using 8th and 9th PC as features')
# plt.scatter(X_PC89[:15, 0], X_PC89[:15, 1], c='r', marker='o', label='Apple')
# plt.scatter(X_PC89[15:, 0], X_PC89[15:, 1], c='y', marker='^', label='Banana')
# plt.legend()
# plt.xlabel('$x_{8}$')
# plt.ylabel('$x_{9}$')
#plt.show()


########################  Using PCA for House Price Prediction
print('##########################################################')
print('##########################################################')
print('########################:-)###############################')
print('##########################################################')
print('##########################################################')



# import "Pandas" library/package (and use shorthand "pd" for the package)
# Pandas provides functions for loading (storing) data from (to) files
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, HTML
import numpy as np
from sklearn.datasets import load_boston
import random


def GetFeaturesLabels(m=10, D=10):
    house_dataset = load_boston()
    # print(house_dataset.feature_names)
    house = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
    x1 = house['TAX'].values.reshape(-1, 1)  # vector whose entries are the tax rates of the sold houses
    x2 = house['AGE'].values.reshape(-1, 1)  # vector whose entries are the ages of the sold houses
    # print(house_dataset)
    # print('house  ################################################################3')
    # print(house)
    x1 = x1[0:m]
    x2 = x2[0:m]
    # print('x1 is', x1)
    # print('x2 is', x2)
    np.random.seed(43)
    Z = np.hstack((x1, x2, np.random.randn(m, n)))
    #print(Z)
    Z = Z[:, 0:D]
    #print('##########################################################')

    y = house_dataset.target.reshape(-1, 1)  # create a vector whose entries are the labels for each sold house
    y = y[0:m]

    return Z, y

#####################Student Task. PCA with Linear Regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

m = 500  # total number of data points
D = 10  # length of raw feature vector

Z, y = GetFeaturesLabels(m, D)  # read in m data points from the house sales database
print('Z is', Z)

## use this features for PCA
Z_pca = Z[:480, :]  # read out feature vectors of first 480 data points

## use this features and labels for linear regression (with transformed features)
Z = Z[480:, :]  # read out feature vectors of last 20 data points
y = y[480:, :]  # read out labels of last 20 data points

# Datasets which will be preprocessed and used with linear regression
Z_train, Z_val, y_train, y_val = train_test_split(Z, y, test_size=0.2, random_state=42)

err_val = np.zeros(D)  # this numpy array has to be used to store the validation
#  errors of linear regression when combined with PCA with n=1,2,..,D

err_train = np.zeros(D)

for n in range(1, D + 1, 1):
    # Create the PCA object and fit
    print('Starting round ', n)
    pca = PCA(n_components=n)
    pca.fit(Z_pca)

    # transform long feature vectors (length D) to short ones (length n)
    X_train = pca.transform(Z_train)
    #print('Z_train is', Z_train)
    #print('X_train is', X_train)
    X_val = pca.transform(Z_val)

### STUDENT TASK ###
#  use X_train,y_train to train linear regression model
#  use resulting model to predict labels for training and validation data w
#  store the predicted labels on training set in numpy array y_pred_train and
#  predicted labels on validation set in numpy array y_pred_val
# YOUR CODE HERE
#raise NotImplementedError()

    LinReg = LinearRegression()
    LinReg = LinReg.fit(X_train, y_train)
    y_pred_train = LinReg.predict(X_train)
    #print('y-pred-train is', y_pred_train)
    y_pred_val = LinReg.predict(X_val)
    #print('y-pred-val is', y_pred_val)

    err_train[n - 1] = np.mean((y_train - y_pred_train) ** 2)  # compute training error
    print('err_train___ is', err_train[n-1])
    #training_error = mean_squared_error(y_train, y_pred_train)
    #print('err_training is', training_error)
    err_val[n - 1] = np.mean((y_val - y_pred_val) ** 2)  # compute validation error
    print('err_val is', err_val[n - 1])

plt.plot([n for n in range(1, D + 1, 1)], err_val, label="validation")
plt.plot([n for n in range(1, D + 1, 1)], err_train, label="training")
plt.xlabel('number of PCs ($n$)')
plt.ylabel(r'error')
plt.legend()
plt.title('validation/training error vs. number of PCs')
plt.show()
