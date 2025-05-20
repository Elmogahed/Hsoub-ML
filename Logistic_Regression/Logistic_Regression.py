import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')
X = df[['Age','Education']]
y = df['VIP']

dfVIP = df[df['VIP'] == 1]
dfNotVIP = df[df['VIP'] == 0]


plt.scatter(dfVIP['Education'],dfVIP['Age'],color='red',marker='x')
plt.scatter(dfNotVIP['Education'],dfNotVIP['Age'],color='blue',marker='x')
plt.legend (["VIP","Not VIP"])
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
b0=round(model.intercept_,2)
b1=round(model.coef_[0],2)
b2=round(model.coef_[1],2)
print('Intercept (b0):', b0)
print('Coefficient (b1):', b1)
print('Coefficient (b2):', b2)

import math


def sigmoid(z):
    return 1 / (1 + pow(math.e, -z))


Age = 69
Education = 12
Boundary = 0.5
y_lr = b0 + b1 * Age + b2 * Education
sig_y = sigmoid(y_lr)
if sig_y > Boundary:
    print("VIP")
else:
    print("Not VIP")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X.values, y)

Age = 60
Education = 12
example = [[Age, Education]]
pred = model.predict(example)
if pred == 1:
    print("VIP")
else:
    print("Not VIP")

p_proba = model.predict_proba(example)
print(p_proba)