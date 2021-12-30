from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
#from IPython import display
import numpy as np


#read in data from the csv file and store it in the numpy array data.
df = pd.read_csv("data.csv")
data = np.array(df)

#display first 5 rows
#display(df.head(5))

X = np.zeros([400, 2])  # read the Cafe customer data into X
cluster_means = np.zeros([2, 2])  # store the resulting clustering means in the rows of this np array
cluster_indices = np.zeros([400, 1])  # store here the resulting cluster indices (one for each data point)
data_colors = ['orangered', 'dodgerblue', 'springgreen']  # colors for data points
centroid_colors = ['red', 'darkblue', 'limegreen']  # colors for the centroids

X = data  # read in cafe costumer data point into numpy array X of shape (400,2)
k_means = KMeans(n_clusters=2, max_iter=100).fit(X)  # apply k-means with k=3 cluster and using 100 iterations
cluster_means = k_means.cluster_centers_  # read out cluster means (centers)
cluster_indices = k_means.labels_  # read out cluster indices for each data point
cluster_indices = cluster_indices.reshape(-1, 1)  # enforce numpy array cluster_indices having shape (400,1)

print('cluster_means')
print(cluster_means)

# code below creates a colored scatterplot

# plt.figure(figsize=(5.75, 5.25))
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

plt.scatter(cluster_means[:, 0], cluster_means[:, 1], marker="x", c='black', s=cent_sz)

plt.show()