transactions = [['Bread', 'Milk'],
                ['Bread', 'Cheese', 'Juice', 'Eggs'],
                ['Milk', 'Cheese', 'Juice', 'Coke' ],
                ['Bread', 'Milk', 'Cheese', 'Juice'],
                ['Bread', 'Milk', 'Cheese', 'Coke']]

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te.fit(transactions)
rows = te.transform(transactions)
print(rows)

import pandas as pd
df_transactions = pd.DataFrame(rows, columns=te.columns_)
print(df_transactions)

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df_transactions, min_support=0.4, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x : len(x))
print(frequent_itemsets)

from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
rules = rules.sort_values(['confidence'], ascending=[False])
print(rules)
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates
rules['antecedent'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequent']= rules['consequents'].apply(lambda a : ','.join(list(a)))
rules['rule_n']  = rules.index
coords = rules[['antecedent', 'consequent', 'rule_n']]
plt.figure(figsize=(4,8))
parallel_coordinates(coords, 'rule_n')
plt.legend([])
plt.grid(True)
plt.show()

import seaborn as sns

matrix = rules.pivot(index = 'antecedent', columns= 'consequent', values='confidence')
print(matrix)

sns.heatmap(matrix, annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

def predict(items, rules, max_results=6):
    preds = rules[rules['antecedents'] == items]
    preds = preds[['consequent', 'confidence']]
    preds.sort_values('confidence', ascending=False)
    return preds[:max_results]

preds = predict({'Cheese', 'Milk'}, rules)
print(preds)




























