import pandas as pd

cars = pd.read_csv("fuel.csv")
print(cars)

cars['class_label'] = cars["drive"].apply(lambda x: 1 if x == 'Front-Wheel Drive' else 0)
print(cars['class_label'])
print(cars)
cars_new = cars[["make", "model", "VClass", "drive", "displ", "comb", "class_label"]]
print(cars_new.head(15))

from sklearn.model_selection import train_test_split
X = cars_new[['displ', 'comb']]
y = cars["class_label"]
# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler()
X = cars_new[['displ', 'comb']]
X_scaled = stscaler.fit_transform(X)
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train.values, y_train)
model_scaled = LogisticRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test.values)
# print(y_pred)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:", precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:", recall)
y_pred_scaled = model_scaled.predict(X_test_scaled)
# print(y_pred_scaled)
cm = metrics.confusion_matrix(y_test, y_pred_scaled)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred_scaled)
print("Accuracy score:", accuracy)
precision = metrics.precision_score(y_test, y_pred_scaled)
print("Precision score:", precision)
recall = metrics.recall_score(y_test, y_pred_scaled)
print("Recall score:", recall)

car = model.predict([[4.4, 16.1]])
if car == 1:
    print("The Car is Front-Wheel Driven")
else:
    print("The Car is Back-Wheel Driven or All-Wheel Driven")

car_1 = model.predict([[2.526, 30.7]])
if car_1 == 1:
    print("The Car is Front-Wheel Driven")
else:
    print("The Car is Back-Wheel Driven or All-Wheel Driven")
























