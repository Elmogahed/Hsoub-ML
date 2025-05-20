import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

df = pd.read_csv('Advertising.csv')
print(df.head())
print(df.isnull().sum())

print(pd.DataFrame(df.isnull().sum(), columns=['Count of null values']).T)

x = df[['TV']]
y = df['Sales']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , train_size=0.8,random_state=50)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.values, y_train)

b0 = round(model.intercept_,2)
b1 = round(model.coef_[0],2)
print('Intercept (b0):', b0)
print('Coefficient (b1):', b1)

Tv_advertisment = [[120]]
predict_one = model.predict(Tv_advertisment)
print("Prediction = ", predict_one[0])

print(Tv_advertisment)

b0 = model.intercept_
b1 = model.coef_[0]
a = Tv_advertisment[0][0]
predict_two = b0 + b1 * a
print("prediction = ", predict_two)

y_predict = model.predict(x_test.values)
print(y_predict)

diff = pd.DataFrame({"Actual": y_test, "Predicted": y_predict})
print(diff.head())

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test,y_predict)
mse = metrics.mean_squared_error(y_test,y_predict)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_predict))

print('mae =',np.round(mae,2))
print('mse =',np.round(mse,2))
print('rmse =',np.round(rmse,2))

sns.regplot(x=y_test, y=y_predict)
plt.show()
