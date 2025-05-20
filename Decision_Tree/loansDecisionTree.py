import pandas as pd

pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv('loan_data_set.csv')
print(df.info())
print(df.describe())

df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1}).astype(int)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(float)
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0}).astype(float)
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype(int)
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0}).astype(float)

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

df = df.drop('Loan_ID', axis=1)

X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']

import matplotlib.pyplot as plt
count_values=y.value_counts()
print(count_values)
labels = count_values.index.to_list()
print(labels)
plt.title('Original classes distribution')
plt.pie(x = count_values, labels = labels, autopct = '%1.1f%%' )
plt.show()

from imblearn.over_sampling import SMOTE
oversample = SMOTE ()
print("Before resampling: ",len(X))
X, y = oversample.fit_resample(X, y)
print("After resampling: ",len(X))
count_values=y.value_counts()
labels = count_values.index.to_list()
plt.title('New classes distribution')
plt.pie(x = count_values, labels = labels, autopct = '%1.1f%%' )
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X))
print(len(X_train))
print(len(X_test))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(100*accuracy_score(y_test, y_pred)))
print('\nPrecision: {:.2f}\n'.format(100*precision_score(y_test, y_pred)))
print('\nRecall: {:.2f}\n'.format(100*recall_score(y_test, y_pred)))
print('\nF1: {:.2f}\n'.format(100*f1_score(y_test, y_pred)))

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
plt.show()