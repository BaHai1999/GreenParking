import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import cv2
import os
from joblib import dump, load

X_train, X_test, y_train, y_test = [], [], [], []
path = os.getcwd()
path = os.path.join(path, 'dataTrainSVM')
for label in os.listdir(path):
    if(label != 'a'):
        label_address = os.path.join(path, label)
        n = len(os.listdir(label_address))
        n_train, dem = int(n*0.8), 0
        for img in os.listdir(label_address):
            x = cv2.imread(os.path.join(label_address, img), 0)
            x = np.reshape(x, x.shape[0]*x.shape[1])
            if(dem < n_train):
                X_train.append(x)
                y_train.append(int(label))
            else:
                X_test.append(x)
                y_test.append(int(label))
            dem += 1
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

parameter_candidates = [
  {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
# clf.fit(X_train, y_train)
# print('Best score:', clf.best_score_)
# print('Best C:',clf.best_estimator_.C)
clf = svm.SVC(C=5)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(classification_report(y_test, y_predict))
dump(clf, 'svm_model_green_parking')