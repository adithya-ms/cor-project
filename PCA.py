# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:30:27 2020

@author: 91782
"""
# Principal Component Analysis (PCA)

# Importing the libraries
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd

from predictors import perform_svm, perform_xgboost, perform_random_forest, perform_naive_bayes, perform_knn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Importing the dataset
def main():
	
	#dataset = pd.read_csv('good_representations.csv')
	dataset = pd.read_csv('inception_representations.csv')
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	
	all_objects = ["Vase", "Teapot","Bottle","Spoon", "Plate", "Mug", "Knife", "Fork", "Flask", "Bowl"]
	#The csv has file path for class labels. This code cleans up this path and adds class name
	for index in range(y.size):
		for obj in all_objects:
			if obj in y[index]:
				y[index] = obj
				break
	
	#Converting labels from strings to integer encoded
	encoder = LabelEncoder()
	y_enc = encoder.fit_transform(y)		
	
	#Feature Scaling before PCA
	sc = StandardScaler()
	X = sc.fit_transform(X)
	
	
	
	#Dimensionality reduction with PCA 
	
	#for GOOD 125 features accounts for 99.9% variance 
	#pca = PCA(n_components=125) 
	
	#for inception 250 features accounts for 99.9% variance
	pca = PCA(n_components=250)   
	X = pca.fit_transform(X)
	
	#Dataset is imbalanced. To account for this, we upsample the representations to get new samples
	sample_count = 100
	
	upsampled_dataset = upsample_all_class(X,y_enc, sample_count)
	
	#The upsampled dataset is split into Dependent and independent variables
	
	X = upsampled_dataset.iloc[:, :-1].values
	y = upsampled_dataset.iloc[:, -1].values
	
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	
	perform_svm(X_train,y_train, X_test, y_test)
	perform_xgboost(X_train,y_train, X_test, y_test)
	perform_random_forest(X_train,y_train, X_test, y_test)
	perform_naive_bayes(X_train,y_train, X_test, y_test)
	perform_knn(X_train,y_train, X_test, y_test)

def upsample_all_class(X,y, sample_count = 100):
	
	df = pd.DataFrame(X)
	
	#Empty dataset to store all samples
	df_upsampled = pd.DataFrame()
	
	df['labels'] = y
	
	for obj_class in np.unique(y):
		#Consider the each object class as minority class and upsample it to 100 samples	
		df_minority = df[df.labels==obj_class]
	
		#Upsample minority class
		df_minority_upsampled = resample(df_minority, 
                                 replace=True,                # sample with replacement
                                 n_samples = sample_count,    # 100 images for each set
                                 random_state=123)            # For reproducible results use seed, else set to None

		#Combine samples with newly upsampled minority class
		df_upsampled = pd.concat([df_upsampled, df_minority_upsampled])
		
	#Display new class counts
	print(df_upsampled.labels.value_counts())	
	return df_upsampled

if __name__ == '__main__':
	main()