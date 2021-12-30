# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

#read in data from the csv file and store it in the numpy array data.
df = pd.read_csv("data.csv")
data = np.array(df)

#display first 5 rows
display(df.head(5))


def plotting(data, centroids=None, clusters=None):
    # this function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}
    # Output: a scatter plot of the data in the clusters with cluster means
    plt.figure(figsize=(5.75, 5.25))
    data_colors = ['orangered', 'dodgerblue', 'springgreen']
    plt.style.use('ggplot')
    plt.title("Data")
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5  # data points alpha
    dt_sz = 20  # marker size for data points
    cent_sz = 130  # centroid sz

    if centroids is None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp, c=data_colors[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp, c=data_colors[0])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=cent_sz, c=centroid_colors[:len(centroids)])
    if centroids is not None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], c=[data_colors[i - 1] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c=centroid_colors[:len(centroids)], s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], c=[data_colors[i - 1] for i in clusters], s=dt_sz, alpha=alp)

    plt.show()


# plot the data
#plotting(data)

from sklearn.cluster import KMeans

X = np.zeros([400, 2])  # read the Cafe customer data into X
cluster_means = np.zeros([2, 2])  # store the resulting clustering means in the rows of this np array
cluster_indices = np.zeros([400, 1])  # store here the resulting cluster indices (one for each data point)
data_colors = ['orangered', 'dodgerblue', 'springgreen']  # colors for data points
centroid_colors = ['red', 'darkblue', 'limegreen']  # colors for the centroids

X = data  # read in cafe costumer data point into numpy array X of shape (400,2)
print('X')
print(X)
k_means = KMeans(n_clusters=3, max_iter=100)  # apply k-means with k=3 cluster and using 100 iterations
k_means = k_means.fit(X)

cluster_means = k_means.cluster_centers_  # read out cluster means (centers)
print('Cluster means ################3')
print(cluster_means)
cluster_indices = k_means.labels_  # read out cluster indices for each data point
print('########################')
print(cluster_indices.shape)
cluster_indices = cluster_indices.reshape(-1, 1)  # enforce numpy array cluster_indices having shape (400,1)


# code below creates a colored scatterplot

# plt.figure(figsize=(6, 5))
# plt.style.use('ggplot')
# plt.title("Data")
# plt.xlabel("feature $x_1$: customers' age")
# plt.ylabel("feature $x_2$: money spent during visit")

alp = 0.5  # data points alpha
dt_sz = 40  # marker size for data points
cent_sz = 130  # centroid sz

# iterate over all cluster indices (minus 1 since Python starts indexing with 0)
for cluster_index in range(3):
    # find indices of data points which are assigned to cluster with index (cluster_index+1)
    indx_1 = np.where(cluster_indices == cluster_index)[0]

    # scatter plot of all data points in cluster with index (cluster_index+1)
    #plt.scatter(X[indx_1, 0], X[indx_1, 1], c=data_colors[cluster_index], s=dt_sz, alpha=alp)

# plot crosses at the locations of cluster means

#plt.scatter(cluster_means[:, 0], cluster_means[:, 1], marker="x", c='black', s=cent_sz)

#plt.show()