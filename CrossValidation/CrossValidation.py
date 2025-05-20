import pandas as pd


df = pd.read_csv('iris.csv')

df['Species'] = df['Species'].replace({
    'Iris-setosa': 0,
    'Iris-virginica': 1,
    'Iris-versicolor': 2
}).astype(int)

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

from sklearn.model_selection import cross_val_score
import sklearn.model_selection
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

scores_accuracy = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(scores_accuracy)
meanScore = scores_accuracy.mean()
print("Accuracy = ", round(meanScore * 100, 2))