import pandas as pd
dataset = pd.read_csv("iPhone.csv")
print(dataset)

print(dataset.describe())

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3]

from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="liblinear")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:",recall)

person1 = sc.transform([[1,21,40000]])
person2 = sc.transform([[1,41,80000]])
person3 = sc.transform([[0,41,40000]])
print("Male aged 21 making $40k will buy iPhone:", classifier.predict(person1))
print("Male aged 41 making $80k will buy iPhone:", classifier.predict(person2))
print("Female aged 41 making $40k will buy iPhone:", classifier.predict(person3))

if classifier.predict(person1) == 0:
    print("Male aged 21 making $40k: Will not purchase iPhone")
else:
    print("Male aged 21 making $40k: Will purchase iPhone")

if classifier.predict(person2) == 0:
    print("Male aged 41 making $80k: Will not purchase iPhone")
else:
    print("Male aged 41 making $80k: Will purchase iPhone")

if classifier.predict(person3) == 0:
    print("Female aged 41 making $40k: Will not purchase iPhone")
else:
    print("Female aged 41 making $40k: Will purchase iPhone")

print("Male aged 21 making $40k will buy iPhone:", classifier.predict_proba(person1))
print("Male aged 41 making $80k will buy iPhone:", classifier.predict_proba(person2))
print("Female aged 41 making $40k will buy iPhone:", classifier.predict_proba(person3))