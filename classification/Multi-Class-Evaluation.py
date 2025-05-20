from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

actual = ['Truck','Truck','Truck','Truck','Truck','Truck','Truck','Truck','Truck','Truck','Truck','Plane','Plane','Plane',
          'Plane','Plane','Plane','Plane','Plane','Plane','Plane','Plane','Plane','Boat','Boat','Boat','Boat','Boat','Boat',
          'Boat','Boat','Boat','Boat','Boat','Boat','Boat']

predicted = ['Truck','Truck','Truck','Truck','Truck','Truck','Truck','Plane','Boat','Boat','Boat','Truck','Truck','Truck',
             'Truck','Truck','Truck','Truck','Truck','Plane','Plane','Boat','Boat','Truck','Truck','Truck','Truck','Truck',
             'Truck','Truck','Truck','Truck','Plane','Plane','Plane','Boat']
for i in range(len(actual)):
    if actual[i] == 'Truck':
        actual[i] = 0
    elif actual[i] == 'Plane':
        actual[i] = 1
    else:
        actual[i] = 2
for i in range(len(predicted)):
    if predicted[i] == 'Truck':
        predicted[i] = 0
    elif predicted[i] == 'Plane':
        predicted[i] = 1
    else:
        predicted[i] = 2

cf_matrix = confusion_matrix(predicted, actual)
print('Confusion Matrix\n', cf_matrix)

sns.heatmap(cf_matrix, annot=True)
plt.show()

from sklearn.metrics import accuracy_score
print('Accuracy: {:.2f}'.format(accuracy_score(actual, predicted)))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(actual,predicted))

print('Micro Precision: {:.2f}'.format(precision_score(actual, predicted, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(actual, predicted, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(actual, predicted, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(actual,predicted, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(actual, predicted, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(actual, predicted, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(actual,predicted, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(actual,predicted, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(actual,predicted, average='weighted')))

from sklearn.metrics import classification_report

print('\nClassification Report\n')
print(classification_report(actual, predicted, target_names=['Truck', 'Plane', 'Boat']))