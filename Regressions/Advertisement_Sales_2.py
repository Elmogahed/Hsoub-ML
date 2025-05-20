import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

df = pd.read_csv('Advertising.csv')
print(df.head())
print(df.columns)

x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=15)
print(x_train.shape)
print(x_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.values, y_train)
print(model.intercept_)
print(model.coef_)

print("The model model is: Y = ",model.intercept_, "+", model.coef_[0], "TV + ", model.coef_[1], "radio + ", model.coef_[2], "newspaper")
advertisment = [[120, 50, 60]]
pred = model.predict(advertisment)
print("prediction= ", pred[0])

y_pred = model.predict(x_test.values)
print(y_pred)

sns.regplot(x = y_test, y = y_pred, color='green')
plt.show()

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print('mae =',np.round(mae,2))
print('mse =',np.round(mse,2))
print('rmse =',np.round(rmse,2))