from sklearn.neighbors import KNeighborsClassifier

X = [[7,7],[7,4],[1,2],[2,2]]
y = [0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

pred = model.predict([[5,6]])

if pred == 0:
    print('This fruit is a pear')
else:
    print('This fruit is a grape')