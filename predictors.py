# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:36:33 2020

@author: 91782
"""
# k-Fold Cross Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def perform_svm(X_train,y_train, X_test, y_test):
	classifier = SVC(kernel = 'rbf', random_state = 0)
	classifier.fit(X_train, y_train)
	
	# Making the Confusion Matrix
	
	plot_confusion_matrix(classifier,X_test,y_test)
	plt.show()

	#y_pred = classifier.predict(X_test)	
	#accuracy_score(y_test, y_pred)
	
	# Applying k-Fold Cross Validation
	
	accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
	print("Accuracy of SVM: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of SVM: {:.2f} %".format(accuracies.std()*100))


def perform_xgboost(X_train,y_train, X_test, y_test):
	
	# XGBoost with GB Decision Trees as classifier
	classifier = XGBClassifier()
	classifier.fit(X_train, y_train)

	# Making the Confusion Matrix
	#y_pred = classifier.predict(X_test)
	#accuracy_score(y_test, y_pred))
	
	plot_confusion_matrix(classifier, X_test, y_test)
	plt.show()
	
	# Applying k-Fold Cross Validation
	accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
	print("Accuracy of XGBoost: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of XGBoost: {:.2f} %".format(accuracies.std()*100))
	
def perform_random_forest(X_train,y_train, X_test, y_test):
	# Random Forest Classification

	# Training the Random Forest Classification model on the Training set
	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
	
	classifier.fit(X_train, y_train)
	# Predicting the Test set results
	
	#y_pred = classifier.predict(X_test)
	
	plot_confusion_matrix(classifier, X_test, y_test)  # doctest: +SKIP
	plt.show()

	# Applying k-Fold Cross Validation
	accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
	print("Accuracy of Random Forest: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of Random Forest: {:.2f} %".format(accuracies.std()*100))

def perform_naive_bayes(X_train,y_train, X_test, y_test):
	#Naive Bayes Classifier
	
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)

	# Predicting the Test set results
	#y_pred = classifier.predict(X_test)
	#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	plot_confusion_matrix(classifier, X_test, y_test)
	plt.show()

	# Applying k-Fold Cross Validation
	accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
	print("Accuracy of NB: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of NB: {:.2f} %".format(accuracies.std()*100))
	
	
def perform_knn(X_train,y_train, X_test, y_test):
	
	classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
	classifier.fit(X_train, y_train)

	# Predicting the Test set results
	#y_pred = classifier.predict(X_test)
	#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	plot_confusion_matrix(classifier, X_test, y_test)  # doctest: +SKIP
	plt.show()

	# Applying k-Fold Cross Validation
	accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
	print("Accuracy of NB: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of NB: {:.2f} %".format(accuracies.std()*100))
	# Training the K-NN model on the Training set

