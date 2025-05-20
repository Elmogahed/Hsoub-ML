from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

actual = ['spam', 'not spam', 'spam', 'not spam', 'spam', 'spam', 'not spam', 'spam', 'not spam', 'not spam']
predicted = ['spam', 'spam', 'spam', 'not spam', 'spam', 'spam', 'not spam', 'not spam', 'not spam', 'spam']

cf_matrix = confusion_matrix(predicted, actual, labels=['not spam', 'spam'])
print('Confusion Matrix: \n', cf_matrix)

TP = cf_matrix[0,0]
TN = cf_matrix[1,1]
FP = cf_matrix[0,1]
FN = cf_matrix[1,0]
print ('TP :{:.2f}'.format(TP))
print ('TN :{:.2f}'.format(TN))
print ('FP :{:.2f}'.format(FP))
print ('FN :{:.2f}'.format(FN))

print ('Accuracy Score :{:.2f}'.format(accuracy_score(actual, predicted)*100))
print ('Precision Score :{:.2f}'.format(precision_score(actual, predicted, pos_label='not spam')*100))
print ('Recall Score :{:.2f}'.format(recall_score(actual, predicted, pos_label='not spam')*100))
print ('F1 Score :{:.2f}'.format(f1_score(actual, predicted, pos_label='not spam')*100))
print ('Specificity :{:.2f}'.format(FP/(FP+TN)*100))

import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
plt.show()
