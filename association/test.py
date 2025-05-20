import pandas as pd

# قراءة البيانات
df = pd.read_csv('Groceries.csv', header=None)
print(df.head())

# تحويل الشكل للصفوف إلى أعمدة (تحويل أفقي)
df = df.T
print(df.head())

# حذف القيم الفارغة من كل عمود وتحويله إلى قائمة
transactions = df.apply(lambda x: x.dropna().tolist())
print(transactions)

# تحويل transactions إلى قائمة من القوائم
transactions = transactions.tolist()
print(transactions)

# تحويل البيانات إلى صيغة مناسبة للأبريوري
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_model = te.fit(transactions)
rows = te_model.transform(transactions)
df_transactions = pd.DataFrame(rows, columns=te_model.columns_)
print(df_transactions.shape)
print(df_transactions.head())

# تطبيق خوارزمية Apriori
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df_transactions, min_support=0.005, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)

# استخراج القواعد باستخدام association rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.55)
rules = rules.sort_values(['confidence'], ascending=False)
print(rules)

# نسخة للرسم البياني حتى لا نؤثر على القيم الأصلية
rules_vis = rules.copy()
rules_vis['antecedents'] = rules_vis['antecedents'].apply(lambda a: ','.join(list(a)))
rules_vis['consequents'] = rules_vis['consequents'].apply(lambda a: ','.join(list(a)))
rules_vis['rule_n'] = rules_vis.index

# رسم القواعد باستخدام parallel_coordinates
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates

coords = rules_vis[['antecedents', 'consequents', 'rule_n']]
plt.figure(figsize=(8,10))
parallel_coordinates(coords, 'rule_n')
plt.legend([])
plt.grid(True)
plt.show()

# رسم مصفوفة heatmap للعلاقات
import seaborn as sns
pivot = rules_vis.pivot(index='antecedents', columns='consequents', values='confidence')
plt.figure(figsize=(8,10))
sns.heatmap(pivot, annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

# دالة التوقع بناءً على عناصر الإدخال
def predict(items, rules, max_results=6):
    preds = rules[rules['antecedents'].apply(lambda x: set(x) == set(items))]
    preds = preds[['consequents', 'confidence']]
    preds = preds.sort_values('confidence', ascending=False)
    return preds.head(max_results)

# تجربة التنبؤ
preds = predict({'yogurt', 'tropical fruit', 'other vegetables'}, rules)
print(preds)
