# -*- coding: utf-8 -*-
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

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
import time
'''
This code implements a meta learning algorith with two layers. The first layer of weak learners consists of a combination of simple classifiers.
The second layer is the meta learning which in this case is XGBoost. XGBoost is a boosting classifier using regression trees as weak learner. 
Cross validation score is considered along with the train and test times,f1 score and a plot to show the enhancement of the meta learning algorithm.
'''

seed=211

# get a list of models to evaluate
def get_models():
	models = dict()
	models['rf'] = RandomForestClassifier(n_estimators = 1000)
	models['et']= ExtraTreesClassifier(n_estimators = 1000, criterion = "entropy",max_depth = 100)
	models['svm'] = SVC(kernel = 'rbf', C= 100, gamma = "scale")
	models['lr'] = LogisticRegression(solver="newton-cg", penalty = "l2", C= 0.1)
	models['knn'] = KNeighborsClassifier(metric= 'euclidean', n_neighbors= 3, weights= 'distance')
	models['stacking'] = get_stacking()
	return models


def evaluate_model(name,model, X_train, y_train, X_test, y_test):
	'''
	Evaluate the given model using cross-validation and print the train and test time
	'''
	
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='f1_micro',
						   cv=cv, n_jobs=-1, error_score='raise')
	train_start=time.time()
	model.fit(X_train,y_train)
	train_stop=time.time()
	test_start = time.time()
	y_pred = model.predict(X_test)
	test_stop = time.time()
	print("train tim:{}\ntest time:{}",train_stop-train_start,test_stop-test_start)
	print(name + "Test Accuracy: " + str(f1_score(y_test, y_pred, average='micro')))
	return scores

def get_stacking():
	'''define the base models and the meta learner as XGBooster. The models are stacked and the final meta learning algorithm is created.
	'''

	level0 = list()
	level0.append(('rf',RandomForestClassifier(n_estimators = 2000)))
	level0.append(('et',ExtraTreesClassifier(n_estimators = 1000, criterion = "entropy",max_depth = 100)))
	level0.append(('svm', SVC(kernel = 'rbf', C= 100, gamma = "scale")))
	level0.append(('lr', LogisticRegression(solver="newton-cg", penalty = "l2", C= 0.1)))
	level0.append(('knn', KNeighborsClassifier(metric= 'euclidean', n_neighbors= 3, weights= 'distance')))	

	# define meta learner model
	level1 = XGBClassifier()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model


def main():
	'''
	The files are read and the target field is encoded since they are currently strings. Demensionality reduction with PCA is performed
	'''


	dataset = pd.read_csv("inception_aug_dataset.csv")
	#dataset = pd.read_csv('inception_aug_dataset.csv')
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
	
	#for GOOD 500 features accounts for 99% variance 
	pca = PCA(n_components=500) 
	
	#for inception 700 features accounts for 99% variance
	#pca = PCA(n_components=700)   
	X = pca.fit_transform(X)
	

	
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
		scores = evaluate_model(name, model, X_train, y_train, X_test, y_test)
		results.append(scores)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	# plot model performance for comparison
	pyplot.boxplot(results * 100, labels=names, showmeans=True)
	pyplot.xlabel("Classifier (Weak and Stacked) ")
	pyplot.ylabel("Accuracy in Percentage")
	pyplot.title("Accuracy of Base and Meta Learners in Stacking")
	pyplot.legend(names, ['Random Forest', 'Extra Trees','SVM', 'Logistic Reg', 'KNN', 'Stacked'])
	pyplot.show()
	


if __name__ == '__main__':
	main()