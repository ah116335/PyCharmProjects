from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import display
import numpy as np
from sklearn.cluster import KMeans

# X = np.zeros([400,2])               # read the Cafe customer data into X
# cluster_means = np.zeros([2,2])     # store the resulting clustering means in the rows of this np array
# cluster_indices = np.zeros([400,1]) # store here the resulting cluster indices (one for each data point)
#
# ### STUDENT TASK ###
# # YOUR CODE HERE
# #raise NotImplementedError()
# X = np.array(pd.read_csv("data.csv"))
# VAR_k_means = KMeans(n_clusters=2, max_iter=20)
# VAR_k_means = VAR_k_means.fit(X)
#
# cluster_means = VAR_k_means.cluster_centers_
# cluster_indices = (VAR_k_means.labels_)
# cluster_indices = cluster_indices.reshape(-1,1)
#
#
# #print(type(cluster_indices))
# #plotting(X,cluster_means,cluster_indices)
# print("The final cluster mean values are:\n",cluster_means)
#
#
#
# assert X.shape == (400, 2), f'numpy array X has wrong shape'
# assert cluster_means.shape == (2, 2), f'numpy array cluster_means has wrong shape'
# assert cluster_indices.shape == (400, 1), f'numpy array cluster indices has wrong shape'
# print('Sanity check tests passed!')
#
#
# print('All tests passed!')


#############################################################3 std task repeaat k means to escape local minima
print('#######################################################3')

print('escape local minima')
print('#######################################################3')

# Student Task
# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import display
import numpy as np


#read in data from the csv file and store it in the numpy array data.
df = pd.read_csv("data.csv")
data = np.array(df)

#display first 5 rows
#display(df.head(5))
min_ind= 0  # store here the index of the repetition yielding smallest clustering error
max_ind= 0  # .... largest clustering error
VAR_k_means= 0

# initializing the array where we collect all cluster assignments
cluster_assignment = np.zeros((50, data.shape[0]),dtype=np.int32)
clustering_err = np.zeros([50,1]) # init numpy array for storing the clustering errors in each repetition
np.random.seed(42)   ###  DO NOT CHANGE THIS LINE !!!

init_means_cluster1 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means
init_means_cluster2 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means
init_means_cluster3 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means

best_assignment = np.zeros((400,1))     # store here the cluster assignment achieving smallest clustering error
worst_assignment = np.zeros((400,1))    # store here the cluster assignment achieving largest clustering error
# YOUR CODE HERE
#raise NotImplementedError()
VAR_cluster_indices = np.zeros([400,1])

for r in range(50):
    #print('Starting round %d' %(r+1))
    i0 = init_means_cluster1[r]
    i1 = init_means_cluster2[r]
    i2 = init_means_cluster3[r]
    k_init = np.zeros([3,2])
    k_init[0]=i0
    k_init[1]=i1
    k_init[2]=i2
    VAR_k_means=KMeans(n_clusters=3, max_iter=10, init=k_init)
    VAR_k_means=VAR_k_means.fit(data)
    VAR_cluster_means = VAR_k_means.cluster_centers_
    VAR_cluster_indices = VAR_k_means.labels_
    clustering_err[r] = (VAR_k_means.inertia_/400)
    cluster_assignment[r] = VAR_cluster_indices

print("clustering_err")
print(clustering_err)
min_ind = (clustering_err.argmin())
max_ind  = (clustering_err.argmax())

print("Cluster assignment with smallest clustering error")
print(np.amin(clustering_err))
print(min_ind)
#print(cluster_assignment[min_ind])
best_assignment = cluster_assignment[min_ind]
best_assignment = best_assignment.reshape(-1,1) # enforce (400,1) shape
print('Best_assignment is')
print(best_assignment)
#plotting(data, clusters = cluster_assignment[min_ind, :])


print("Cluster assignment with largest clustering error")
#plotting(data, clusters = cluster_assignment[max_ind,:])
print(np.amax(clustering_err))
print(max_ind)
#print(cluster_assignment[max_ind])
worst_assignment = cluster_assignment[max_ind]
worst_assignment = worst_assignment.reshape(-1,1) # enforce (400,1) shape
#print('Worst_assignment is')
#print(worst_assignment)

assert all(best_assignment) == 0, 'You have to assign value for best_assignment '
assert all(worst_assignment) == 0, 'You have to assign value for worst_assignment '
assert best_assignment.shape[0] == 400, 'incorrect cluster labels for minimal clustering error'
assert worst_assignment.shape[0] == 400, 'incorrect cluster labels for maximal clustering error'
print('Sanity check tests passed!')

print('All tests passed!')

print('#######################################################3')
print('END / escape local minima')
print('#######################################################3')


################################################## Student Task

df = pd.read_csv("data.csv")
data = np.array(df)
data_num = data.shape[0]
err_clustering = np.zeros([8,1])
VAR_k_means= 0

# YOUR CODE HERE
#raise NotImplementedError()


for r in range(8):
    #print('Starting round %d' %(r+1))
    k_index=r+1
    VAR_k_means=KMeans(n_clusters=k_index, max_iter=100, n_init=1)
    VAR_k_means=VAR_k_means.fit(data)
    VAR_k_means.cluster_centers_
    err_clustering[r] = (VAR_k_means.inertia_/400)
    # print('cluster centers')
    # print(VAR_k_means.cluster_centers_)

print(err_clustering)

fig=plt.figure(figsize=(8,6))
plt.plot(range(1,9),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
plt.show()

################################################## Student Task DBSCAN
# implementing DBSCAN

print('############################### Starting DBSCAN')

from sklearn.cluster import DBSCAN

### STUDENT TASK ###

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000,2))
clust2 = np.random.normal(15, 3, (1000,2))
clust3 = np.random.multivariate_normal([17,3], [[1,0],[0,1]], 1000)
clust4 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], 1000)

dataset1 = np.concatenate((clust1, clust2, clust3, clust4))
print(dataset1)
# we take the first array as the second array has the cluster labels
dataset2 = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)[0]

# YOUR CODE HERE
#raise NotImplementedError()

# The DBSCAN implementation fit_predict(self, X[, y, sample_weight]) returns cluster labels.

dbscan_dataset1 = np.zeros([4000,1])
dbscan_dataset2 = np.zeros([1000,1])
dataset1_noise_points = None
dataset2_noise_points = None
eps_ds1 = 1
eps_ds2 = 0.1
min_samples = 5
print(dataset1)
dbscan_dataset1 = DBSCAN(eps=eps_ds1, min_samples=min_samples, metric='euclidean').fit_predict(dataset1)
dbscan_dataset2 = DBSCAN(eps=eps_ds2, min_samples=min_samples, metric='euclidean').fit_predict(dataset2)

print('dbscan_dataset1')
print(dbscan_dataset1)
dbscan_dataset1 = dbscan_dataset1.reshape(-1,1)
dbscan_dataset2 = dbscan_dataset2.reshape(-1,1)
dataset1_noise_points = (dbscan_dataset1 == -1).sum()
dataset2_noise_points = (dbscan_dataset2 == -1).sum()


print('dbscan_dataset1 *************************')
print(dbscan_dataset1[dbscan_dataset1 < 0])
# print('dbscan_dataset2')
# print(dbscan_dataset2)

print('dbscan_dataset1.shape')
print(dbscan_dataset1.shape)
print('Dataset1:')
print("Number of Noise Points: ",dataset1_noise_points," (",len(dbscan_dataset1),")",sep='')
print('Dataset2:')
print("Number of Noise Points: ",dataset2_noise_points," (",len(dbscan_dataset2),")",sep='')

#cluster_plots(dataset1, dataset2, dbscan_dataset1, dbscan_dataset2)

# This cell is used for grading the assignment, please do not delete it!
assert dbscan_dataset1.shape[0] == 4000, 'Shape of dbscan_dataset1 is wrong.'
assert dbscan_dataset1.shape[1] == 1, 'Shape of dbscan_dataset1 is wrong.'
assert dbscan_dataset2.shape[0] == 1000, 'Shape of dbscan_dataset1 is wrong.'
assert dbscan_dataset2.shape[1] == 1, 'Shape of dbscan_dataset1 is wrong.'
assert dataset1_noise_points < 50, 'Number of noise points in dataset 1 should be less than 50.'
assert dataset2_noise_points < 5, 'Number of noise points in dataset 2 should be less than 5.'
print('Sanity check tests passed!')


print('All tests passed!')

bc = np.zeros((3,2))
cd = np.zeros([3,2])

print(bc)
print(cd)
