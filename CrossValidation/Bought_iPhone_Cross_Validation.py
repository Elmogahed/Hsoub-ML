import pandas as pd

dataset = pd.read_csv("iPhone.csv")
print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()

X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import cross_val_score
def cv_comparison_classification(models, X, y, cv):
    cv_df = pd.DataFrame()
    for model in models:
        acc = cross_val_score(model, X,y,scoring='accuracy', cv=cv)
        acc_avg = round(acc.mean(),3)
        cv_df[str(model)] = [acc_avg]
        cv_df.index = ['Accuracy']

    return cv_df

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

mlr_d = DecisionTreeClassifier(criterion='entropy')
mlr_g = GaussianNB()
mlr_reg = LogisticRegression()
mlr_k = KNeighborsClassifier()
mlr_svc = svm.SVC(kernel='rbf')

models = [mlr_d, mlr_g, mlr_reg, mlr_k, mlr_svc]
comp_df = cv_comparison_classification(models, X, y, 5)
pd.set_option('display.max_columns', None)
print(comp_df)
