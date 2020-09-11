import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

train, test = train_test_split(dataset, test_size=0.2)

train, validate = train_test_split(train, test_size=0.25)

train_X = train.drop('species', axis=1)

train_y = train['species']

validate_X = validate.drop('species', axis=1)

validate_y = validate['species']

values = []

accuracies = []

results = {}

for value1 in range(1, 10):
  for value2 in range(1, 10):

    temp_classifier = DecisionTreeClassifier(max_depth = value1, min_samples_leaf = value2)

    temp_classifier.fit(train_X, train_y)

    predictions_train = temp_classifier.predict(train_X)

    predictions_validate = temp_classifier.predict(validate_X)

    values.append((value1, value2))

    train_acc = accuracy_score(train_y, predictions_train)

    val_acc = accuracy_score(validate_y, predictions_validate)

    accuracies.append((train_acc, val_acc))

results = zip(values, accuracies)

for num, score in results:
  print('For max_value, min_samples_leaf = ' + str(num) + ' the accuracies for training and validating are ' + str(score))

#After analyzing results dictionary, manually input best max_depth and min_samples_leaf values
classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)

classifier.fit(train_X, train_y)

plt.figure(figsize=[7, 7]) #Sets in [x, x] square
plot_tree(classifier, 
          feature_names=train.columns, 
          class_names=classifier.classes_)
plt.show()

predictions = classifier.predict(train_X)
print('----- Training -----')
print('Accuracy:', accuracy_score(train_y, predictions))
print('Precision:', list(zip(classifier.classes_, precision_score(train_y, predictions, average=None))))
print('Recall:', list(zip(classifier.classes_, recall_score(train_y, predictions, average=None))))

predictions = classifier.predict(validate_X)
print('\n----- Validation -----')
print('Accuracy:', accuracy_score(validate_y, predictions))
print('Precision:', list(zip(classifier.classes_, precision_score(validate_y, predictions, average=None))))
print('Recall:', list(zip(classifier.classes_, recall_score(validate_y, predictions, average=None))))

test_X = test.drop('species', axis=1)
test_y = test['species']

predictions = classifier.predict(test_X)
print('\n----- Testing -----')
print('Accuracy:', accuracy_score(test_y, predictions))
print('Precision:', list(zip(classifier.classes_, precision_score(test_y, predictions, average=None))))
print('Recall:', list(zip(classifier.classes_, recall_score(test_y, predictions, average=None))))