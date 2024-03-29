from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
import sklearn

cancer = datasets.load_breast_cancer()

# print("Features: ", cancer.feature_names)
# print("Labels: ", cancer.target_names)
x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
classes = ['malignant','bengain']

#k lenear applying 
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)

#https://techwithtim.net/tutorials/machine-learning-python/svm-p-3-implementation/
