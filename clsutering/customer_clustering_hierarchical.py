import pandas as pd

df = pd.read_csv('shopping-data.csv')
X = df.drop(columns=["CustomerID"])
print(X.describe())

from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler(feature_range=(1,99))
data_scaler.fit(X)
TX=data_scaler.transform(X)
print(TX)
X['Income']=TX[:,0]
X['Score']=TX[:,1]

from scipy.cluster.hierarchy import dendrogram, linkage
Z1 = linkage(X, method='single', metric='euclidean')
Z2 = linkage(X, method='complete', metric='euclidean')
Z3 = linkage(X, method='average', metric='euclidean')
Z4 = linkage(X, method='ward', metric='euclidean')

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.subplot(2,2,1), dendrogram(Z1), plt.title('Single')
plt.subplot(2,2,2), dendrogram(Z2), plt.title('Complete')
plt.subplot(2,2,3), dendrogram(Z3), plt.title('Average')
plt.subplot(2,2,4), dendrogram(Z4), plt.title('Ward')
plt.show()

from sklearn.cluster import AgglomerativeClustering
Z1 = AgglomerativeClustering(n_clusters=5, linkage='ward')
Z1.fit_predict(X)
print(Z1.labels_)

import numpy as np
import matplotlib.pyplot as plt

X1 = X["Income"]
X2 = X["Score"]

plt.scatter(X1, X2, c=Z1.labels_, cmap='rainbow')

# حساب مركز كل مجموعة لإضافة نقطة ورقم المجموعة في المركز
unique_labels = np.unique(Z1.labels_)
centroids = []
for label in unique_labels:
    centroid = np.mean(np.array([X1[Z1.labels_ == label], X2[Z1.labels_ == label]]).T, axis=0)
    centroids.append(centroid)

# Plot cluster centers
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color='black')
    plt.annotate(
        i,
        xy=centroid,
        textcoords='offset points',
        xytext=(10, 10)
    )
plt.show()

cluster_map = pd.DataFrame()
cluster_map['CustomerID'] = df['CustomerID']
cluster_map['cluster'] = Z1.labels_
print(cluster_map[cluster_map.cluster == 1])