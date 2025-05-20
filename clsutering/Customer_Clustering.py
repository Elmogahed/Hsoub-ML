import pandas as pd

df = pd.read_csv('shopping-data.csv')
X = df.drop(columns=['CustomerID'])
print(X.describe())

from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler(feature_range=(1,99))
data_scaler.fit(X)
TX = data_scaler.transform(X)
X['Income'] = TX[:,0]
X['Score'] = TX[:,1]
print(X)

from sklearn.cluster import KMeans
X = X[['Income', 'Score']]
from sklearn.metrics import silhouette_score
range_n_clusters = [2,3,4,5,6,7,8]
max_silhouette_score = 0
best_n_clusters = 0

for n_clusters in range_n_clusters:
    model = KMeans(n_clusters = n_clusters)
    model.fit(X)
    silhouette_avg = silhouette_score(X,model.labels_)
    if silhouette_avg>max_silhouette_score:
        max_silhouette_score=silhouette_avg
        best_n_clusters=n_clusters
    print(
    "For n_clusters =",
    n_clusters,
    "The average silhouette_score is :",
    round(silhouette_avg,2),
    )
print("Best n_cluster= ",best_n_clusters, " with silhouette score= ", round(max_silhouette_score,2))

n_clusters = best_n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)
import matplotlib.pyplot as plt
X1 = X['Income']
X2 = X['Score']

plt.scatter(X1, X2, c=kmeans.labels_, cmap='rainbow')

XCenters = kmeans.cluster_centers_[:,0]
YCenters = kmeans.cluster_centers_[:,1]
plt.scatter(XCenters ,YCenters, color='black')

for i in range(best_n_clusters):
    plt.annotate(
        i,
        xy=(XCenters[i], YCenters[i]),
        textcoords='offset points',
        xytext=(10, 10))

plt.show()

culster_map = pd.DataFrame()
culster_map['CustomerID'] = df['CustomerID']
culster_map['cluster'] = kmeans.labels_
print(culster_map[culster_map.cluster == 2])













