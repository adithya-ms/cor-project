import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix

from mlens.ensemble import SuperLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

seed=211

def plot_learning_curve(train_sizes, train_scores, test_scores, fit_times):
    
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Traing score vs Validation")

    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def main():
	
	#dataset = pd.read_csv('inception_aug_dataset.csv')
	dataset = pd.read_csv('good_aug_dataset.csv')
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
	
	#for GOOD 230 features accounts for 97% variance 
	#pca = PCA(n_components=230) 
	
	#for inception 500 features accounts for 99.9% variance
	pca = PCA(n_components=500)   
	X = pca.fit_transform(X)	

	
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size = 0.2, random_state = 0)

	perform_ensemble_adaboost(X_train, y_train, X_test, y_test)


def perform_ensemble_adaboost(X_train, y_train, X_test, y_test):

	all_objects = ["Vase", "Teapot","Bottle","Spoon", "Plate", "Mug", "Knife", "Fork", "Flask", "Bowl"]

	ensemble = SuperLearner(folds=10,
    	random_state=seed,
    	verbose=2,
    	backend="multiprocessing",
    	scorer=accuracy_score)

	layer_1 = [SVC(kernel='linear', C=8)]
	ensemble.add(layer_1)

	# 95.50

	"""Make plots of learning curve"""

	ensemble.add_meta(AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=5, min_samples_leaf=8)))

	ensemble.fit(X_train, y_train)

	import time

	start = time.time()
	
	yhat = ensemble.predict(X_test)

	accuracies = cross_val_score(ensemble, X_test, y_test, cv=10, scoring="accuracy")

	print("Accuracy of Adaboost: {:.2f} %".format(accuracies.mean()*100))
	print("Standard Deviation of Adaboost: {:.2f} %".format(accuracies.std()*100))

if __name__ == '__main__':
	main()