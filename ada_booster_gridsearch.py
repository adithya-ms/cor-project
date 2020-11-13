import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

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
    return df_upsampled 



def cv_score_model(mod, X_test, y_test, folds, scoring):
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    cv_estimate = cross_val_score(mod, X_test, y_test, cv=cv, scoring=scoring, n_jobs=4)
    return np.mean(cv_estimate), np.std(cv_estimate)


def fill_results_df(mod_list, name_list, scoring_list, X_train, X_test, y_train, y_test, folds):
    
    results = pd.DataFrame(index=name_list)
    for score in scoring_list:
        sc_mean = '{}_mean'.format(score)
        sc_std = '{}_std'.format(score)
        for name, model in zip(name_list, mod_list):
            print (name)
            model = model.fit(X_train, y_train)
            mean, std = cv_score_model(model, X_test, y_test, folds, score)
            results.loc[name, sc_mean] = mean
            results.loc[name, sc_std] = std
    
    return results

def ada_base_hyperparameter_tuning(ada_base):

    cv=StratifiedKFold(n_splits=10, shuffle=True)

    param_grid={'n_estimators' :[50, 100, 250, 500, 750, 1000],
                'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}

    base_grid = GridSearchCV(estimator=base,
                        param_grid=param_grid,
                        cv=cv,
                        scoring='accuracy',
                        return_train_score=True,
                        n_jobs=4,
                        verbose=1)


    base_grid.fit(X, y)
    base_best_mod = base_grid.best_estimator_

    print('For Adaboost with default decision stump as base estimator\n')
    print('Best GridSearchCV Score roc_auc {}'.format(base_grid.best_score_))
    print('Hyperparameters                 Values')
    print('n_estimators:                    {}'.format(base_grid.best_estimator_.n_estimators))
    print('learning_rate:                   {}'.format(base_grid.best_estimator_.learning_rate))


def ada_deci_hyperparameter_tuning(ada_deci, X, y):
    cv=StratifiedKFold(n_splits=10, shuffle=True)

    param_grid = {'base_estimator__max_depth' :[1, 2, 5],
                  'base_estimator__min_samples_split' :[2, 3 ,5],
                  'base_estimator__min_samples_leaf' :[2, 3, 5 ,10],
                  'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],
                  'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}


    deci_grid = GridSearchCV(estimator=ada_deci,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='accuracy',
                            return_train_score=True,
                            n_jobs=4,
                            verbose=1)

    deci_grid.fit(X, y)

    print('For Adaboost with decision tree as base estimator\n')
    print('Best GridSearchCV roc_auc Score {}'.format(deci_grid.best_score_))
    print('Hyperparameters                   Values')
    print('base_estimator__max_depth:          {}'.format(deci_grid.best_estimator_.base_estimator.max_depth))
    print('base_estimator__min_samples_split:  {}'.format(deci_grid.best_estimator_.base_estimator.min_samples_split))
    print('base_estimator__min_samples_leaf:   {}'.format(deci_grid.best_estimator_.base_estimator.min_samples_leaf))
    print('n_estimators:                       {}'.format(deci_grid.best_estimator_.n_estimators))
    print('learning_rate:                      {}'.format(deci_grid.best_estimator_.learning_rate))


def ada_extr_hyperparameter_tuning(ada_extr, X, y):

    cv=StratifiedKFold(n_splits=10, shuffle=True)

    param_grid = {'base_estimator__criterion' :['gini', 'entropy'],
                  'base_estimator__max_depth' :[1, 2, 5],
                  'base_estimator__min_samples_split' :[2, 3 ,5],
                  'base_estimator__min_samples_leaf' :[2, 3, 5 ,10],
                  'n_estimators' :[10, 50, 100, 250, 500, 750],
                  'learning_rate' :[0.001, 0.01, 0.1, 1]}



    extr_grid = GridSearchCV(estimator=ada_extr,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='accuracy',
                            return_train_score=True,
                            n_jobs=-1,
                            verbose=1)

    extr_grid.fit(X, y)

    print('For Adaboost with extra tree as base estimator\n')
    print('Best GridSearchCV accuracy Score {}'.format(extr_grid.best_score_))
    print('Hyperparameters                   Values')
    print('base_estimator__criterion:          {}'.format(extr_grid.best_estimator_.base_estimator.criterion))
    print('base_estimator__max_depth:          {}'.format(extr_grid.best_estimator_.base_estimator.max_depth))
    print('base_estimator__min_samples_split:  {}'.format(extr_grid.best_estimator_.base_estimator.min_samples_split))
    print('base_estimator__min_samples_leaf:   {}'.format(extr_grid.best_estimator_.base_estimator.min_samples_leaf))
    print('n_estimators:                       {}'.format(extr_grid.best_estimator_.n_estimators))
    print('learning_rate:                      {}'.format(extr_grid.best_estimator_.learning_rate))


def ada_svml_hyperparameter_tuning(ada_svml, X, y):

    cv=StratifiedKFold(n_splits=10, shuffle=True)

    param_grid = {'base_estimator__C' :[0.01, 0.1, 1, 10, 50, 100, 500, 1000],
                  'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],
                  'learning_rate' :[0.001, 0.01, 0.1, 1]}



    svml_grid = GridSearchCV(estimator=ada_svml,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='accuracy',
                            return_train_score=True,
                            n_jobs=-1,
                            verbose=1)

    svml_grid.fit(X, y)
    svml_best_mod = svml_grid.best_estimator_


    print('For Adaboost with linear kernal SVM as base estimator\n')
    print('Best GridSearchCV roc_auc Score {}'.format(svml_grid.best_score_))
    print('Hyperparameters           Values')
    print('base_estimator__C:          {}'.format(svml_grid.best_estimator_.base_estimator.C))
    print('n_estimators:               {}'.format(svml_grid.best_estimator_.n_estimators))
    print('learning_rate:              {}'.format(svml_grid.best_estimator_.learning_rate))







def main():
    
    #dataset = pd.read_csv('good_representations_aug.csv')
    dataset = pd.read_csv('inception_representations_aug.csv')
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
    #pca = PCA(n_components=230) 
    
    #for inception 500 features accounts for 99.9% variance
    pca = PCA(n_components=500)   
    X = pca.fit_transform(X)
    
    #Dataset is imbalanced. To account for this, we upsample the representations to get new samples
    sample_count = 100
    
    #The upsampled dataset is split into Dependent and independent variables
    
    X = upsampled_dataset.iloc[:, :-1].values
    y = upsampled_dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    ada_base = AdaBoostClassifier()
    ada_deci = AdaBoostClassifier(DecisionTreeClassifier())
    ada_extr = AdaBoostClassifier(ExtraTreeClassifier())
    ada_logr = AdaBoostClassifier(LogisticRegression())
    ada_svml = AdaBoostClassifier(SVC(probability=True , kernel='linear'))

    models = [ada_base, ada_deci, ada_extr, ada_logr, ada_svml]
    model_names = ['Base', 'DecisonTree', 'ExtraTree', 'LogisticRegression', 'SVM']

    s = ['accuracy']
    r = fill_results_df(models, model_names, s, X_train, X_test, y_train, y_test, 10)
    print('Results from untuned classifiers', r)


    # Find best parameters

    base = clone(ada_base)
    ada_base_hyperparameter_tuning(base, X, y)

    deci = clone(ada_deci)
    ada_deci_hyperparameter_tuning(deci, X, y)

    extr = clone(ada_extr)
    ada_extr_hyperparameter_tuning(extr, X, y)

    svml = clone(ada_svml)
    ada_svml_hyperparameter_tuning(svml, X, y)

    logr = clone(ada_logr)
    ada_logr_hyperparameter_tuning(logr, X, y)

    


if __name__ == '__main__':
    main()


