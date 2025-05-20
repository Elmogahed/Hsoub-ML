import pandas as pd

df = pd.read_csv('apples_and_oranges.csv')
print(df)

import matplotlib.pyplot as plt
plt.xlabel('Weight')
plt.ylabel('Size')
dforange = df.query("Class =='orange'")
dfapple = df.query("Class =='apple'")
plt.scatter(dforange['Weight'], dforange['Size'], color="red", marker='*')
plt.scatter(dfapple['Weight'], dfapple['Size'], color="blue", marker='*')
plt.show()

from sklearn.model_selection import train_test_split
X = df[['Weight','Size']]
y = df['Class']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train.values, y_train)

y_pred=model.predict(X_test.values)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(100*accuracy_score(y_test, y_pred)))
print('\nPrecision: {:.2f}\n'.format(100*precision_score(y_test, y_pred)))
print('\nRecall: {:.2f}\n'.format(100*recall_score(y_test, y_pred)))
print('\nF1: {:.2f}\n'.format(100*f1_score(y_test, y_pred)))

Weight=62
Size=3
Classpred = model.predict([[Weight,Size]])
Classpred = le.inverse_transform(Classpred)
print(Classpred)

Weight=74
Size=5.3
Classpred = model.predict([[Weight,Size]])
Classpred = le.inverse_transform(Classpred)
print(Classpred)