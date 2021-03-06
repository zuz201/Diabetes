# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#dataset import

dataset = pd.read_csv('diabetes.txt')

#data preprocessing
dataset.info()

dataset.groupby('Outcome').size()
sns.countplot(dataset['Outcome'])

dataset.head()
description = dataset.describe()


def plot_outliers(data):
    
    columns = dataset.columns

    for col in columns[:-1]:
        plt.figure()
        sns.boxplot(x = data[col])



plot_outliers(dataset)

#splitting dataset into train and testset

from sklearn.model_selection import train_test_split

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train_lr, X_test_lr = X_train, X_test
y_train_lr, y_test_lr = y_train, y_test

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_lr = sc_X.fit_transform(X_train_lr)
X_test_lr = sc_X.transform(X_test_lr)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
class_lr = LogisticRegression(random_state = 0)
class_lr.fit(X_train_lr, y_train_lr)

#Prediction
y_pred_lr = class_lr.predict(X_test_lr)
#Confusion Matrix
from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score
    
def model_evaluation(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    
    ass = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return cm, aps, ps, rs, ass, f1s, class_report
    
cm_lr, aps_lr, ps_lr, rs_lr, ass_lr, f1s_lr, class_report_lr = model_evaluation(y_test_lr, y_pred_lr)

#kN-NN
from sklearn.neighbors import KNeighborsClassifier
class_knn = KNeighborsClassifier(n_neighbors = 5,  metric = 'minkowski', p = 2)
class_knn.fit(X_train_lr, y_train_lr)

y_pred_knn = class_knn.predict(X_test_lr)

cm_knn, aps_knn, ps_knn, rs_knn, ass_knn, f1s_knn, class_report_knn = model_evaluation(y_test_lr, y_pred_knn)

#SVM
from sklearn.svm import SVC
class_svm = SVC(kernel = 'linear', random_state = 0)
class_svm.fit(X_train_lr, y_train_lr)
y_pred_svm = class_svm.predict(X_test_lr)

cm_svm, aps_svm, ps_svm, rs_svm, ass_svm, f1s_svm, class_report_svm = model_evaluation(y_test_lr, y_pred_svm)

#Decision tree
from sklearn.tree import DecisionTreeClassifier
class_dt = DecisionTreeClassifier(max_depth = 5, criterion = 'gini', random_state = 0)
class_dt.fit(X_train_lr, y_train_lr)
y_pred_dt = class_dt.predict(X_test_lr)

cm_dt, aps_dt, ps_dt, rs_dt, ass_dt, f1s_dt, class_report_dt = model_evaluation(y_test_lr, y_pred_dt)


    
    

   
