import pandas as pd

df = pd.read_csv('sales.csv')
print(df)

from sklearn.linear_model import LinearRegression

x = df[['Weeks']]
y = df['Total']

model = LinearRegression()

model.fit(x, y)

print('Intercept:', round(model.intercept_,2))
print('Coefficients:', round(model.coef_[0],2))
