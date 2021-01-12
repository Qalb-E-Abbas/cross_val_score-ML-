# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 00:13:10 2021

@author: Qalbe
"""

# I will find score of these three
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#load dataset
from sklearn.datasets import load_digits
digits = load_digits()

#split in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)



#cross_val_score will be used to discover which model fits best
from sklearn.model_selection import cross_val_score


cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)


cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)

cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)