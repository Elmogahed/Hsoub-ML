import pandas as pd

df = pd.read_csv('movies_recommendation_data.csv')
print(df)

X = df[['Biography','Drama','Thriller','Comedy','Crime','Mystery','History']]

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors()
model.fit(X.values)

Movie = [[0,1,1,0,1,0,0]]

neighbors = model.kneighbors(Movie, 5)
print(neighbors)

for index in neighbors[1]:
    print(df.iloc[index, 1])