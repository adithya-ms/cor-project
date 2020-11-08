# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:27:53 2020

@author: Adithya
"""
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

 
from xgboost import XGBClassifier


from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

seed=211

# get a list of models to evaluate
def get_models():
	models = dict()
	models['rf'] = RandomForestClassifier()
	#models['dt'] = DecisionTreeClassifier()
	#models['ada']= AdaBoostClassifier()
	#models['gbm']= GradientBoostingClassifier()
	models['et']= ExtraTreesClassifier()
	models['svm'] = SVC()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	#models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='f1_micro',
						   cv=cv, n_jobs=-1, error_score='raise')
	return scores

def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = XGBClassifier(verbosity=0)
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model


def main():
	
	dataset = pd.read_csv("inception_aug_dataset.csv")
	#dataset = pd.read_csv('inception_representations.csv')
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
	pca = PCA(n_components=700) 
	
	#for inception 250 features accounts for 99.9% variance
	#pca = PCA(n_components=250)   
	X = pca.fit_transform(X)
	
	#Dataset is imbalanced. To account for this, we upsample the representations to get new samples
	sample_count = 100
	
	#upsampled_dataset = upsample_all_class(X,y_enc, sample_count)
	
	#The upsampled dataset is split into Dependent and independent variables
	
	X_df = pd.DataFrame(X)
	y_df = pd.DataFrame(y_enc)
	X_df = X_df.iloc[:, :-1].values
	y_df = y_df.iloc[:, -1].values
	
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.2, random_state = 0)
	# Defining our estimator, the algorithm to optimize
	models = get_models()
	# evaluate the models and store results
	results, names = list(), list()
	for name, model in models.items():
		scores = evaluate_model(model, X_train, y_train)
		results.append(scores)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	# plot model performance for comparison
	pyplot.boxplot(results, labels=names, showmeans=True)
	pyplot.show()
	


if __name__ == '__main__':
	main()
