import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets 
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data
y = iris.target 

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Gaussian Naive Bayes 
gnb = GaussianNB()
#training model 
gnb.fit(X_train, y_train)
#making predictions
gnb_pred = gnb.predict(X_test)
#print accuracy
print('Accuracy of Gaussian Naive Bayes: ', accuracy_score(y_test, gnb_pred))
#print other performance matrics
print("Precision of Gaussian Naive Bayes: ", 
      precision_score(y_test, gnb_pred, average='weighted'))
print("Recall of Gaussian Naive Bayes: ", 
      recall_score(y_test, gnb_pred, average='weighted'))
print("F1-Score of Gaussian Naive Bayes: ",
      f1_score(y_test, gnb_pred, average='weighted'))


#Decision Tree Classifier 
dt = DecisionTreeClassifier(random_state=0)
#training model 
dt.fit(X_train, y_train)
#making prediction 
dt_pred = dt.predict(X_test)
#print accuracy 
print("Accuracy of Decision Tree Classifier: ", 
      accuracy_score(y_test, dt_pred))
# print other performance metrics
print("Precision of Decision Tree Classifier: ",
      precision_score(y_test, dt_pred, average='weighted'))
print("Recall of Decision Tree Classifier: ",
      recall_score(y_test, dt_pred, average='weighted'))
print("F1-Score of Decision Tree Classifier: ",
      f1_score(y_test, dt_pred, average='weighted'))

    
#Support Vector Machine
svm_clf = svm.SVC(kernel='linear') #Linear kernel 
#training model 
svm_clf.fit(X_train, y_train)
#making prediction 
svm_clf_pred = svm_clf.predict(X_test)
# print the accuracy
print("Accuracy of Support Vector Machine: ",
      accuracy_score(y_test, svm_clf_pred))
# print other performance metrics
print("Precision of Support Vector Machine: ",
      precision_score(y_test, svm_clf_pred, average='weighted'))
print("Recall of Support Vector Machine: ",
      recall_score(y_test, svm_clf_pred, average='weighted'))
print("F1-Score of Support Vector Machine: ",
      f1_score(y_test, svm_clf_pred, average='weighted'))
