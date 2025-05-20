import pandas as pd

df = pd.read_csv('Groceries.csv', header=None)
print(df.head())

df = df.T
print(df.head())

transactions = df.apply(lambda x: x.dropna().tolist())
print(transactions)

transactions = transactions.tolist()
print(transactions)

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_model = te.fit(transactions)
rows = te_model.transform(transactions)
df_transactions = pd.DataFrame(rows, columns=te_model.columns_)
print(df_transactions.shape)
print(df_transactions.head())

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df_transactions, min_support=0.005, use_colnames=True)
frequent_itemsets['length']  = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)

from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.55)
rules = rules.sort_values(['confidence'], ascending=False)
print(rules)
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates

rules['antecedent']  = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequent'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
rules['rule_n'] = rules.index
coords = rules[['antecedent', 'consequent', 'rule_n']]
plt.figure(figsize=(8,10))
parallel_coordinates(coords, 'rule_n')
plt.legend([])
plt.grid(True)
plt.show()

import seaborn as sns
pivot = rules.pivot(index = 'antecedent', columns='consequent', values='confidence')
plt.figure(figsize=(8,10))
sns.heatmap(pivot, annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

def predict(items, rules, max_results=6):
    preds = rules[rules['antecedent'].apply(lambda x : set(x) == set(items))]
    preds= preds[['consequent', 'confidence']]
    preds.sort_values('confidence', ascending=False)
    return preds[:max_results]

preds = predict({'yogurt', 'tropical fruit', 'other vegetables'}, rules)
print(preds)













