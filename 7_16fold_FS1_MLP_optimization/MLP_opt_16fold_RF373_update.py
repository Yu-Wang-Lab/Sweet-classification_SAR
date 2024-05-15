# Set work directory
import os
os.chdir("/Users/Zoey/Library/CloudStorage/OneDrive-UniversityofFlorida/PhD_Coding/25_MLP_opt_New")
os.getcwd()

# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
sw = pd.read_csv ('SW_904_zscaled_RF373.csv', sep=',',encoding='utf-8')
ns = pd.read_csv ('NS_318_zscaled_RF373.csv', sep=',',encoding='utf-8')
sw = sw.dropna()
ns = ns.dropna()
    
# features names
feat = list(sw)
feat = feat[1:] # feature name
print(feat)
    
# convert it to numpy array
sw = sw.values
ns = ns.values
    
# features and labels
swl = np.zeros((len(sw),))
swf = sw[:,1:]
nsl = np.ones((len(ns),))
nsf = ns[:,1:]

wdata = np.concatenate((swf,nsf))
est_labels = np.zeros((len(wdata),))
wlabel = np.concatenate((swl,nsl))
 
# inset train_test split
X = wdata
y = wlabel
# Splitting the datafram into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# using the train test split function to generate training set and test set
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

# cross validation and apply multiple machine learning techniques
cv = StratifiedKFold(n_splits = 16, random_state = 42, shuffle = True) # k-fold cross validation
stratified_16folds = cv.split(X_train, y_train) #Fifteen-16 folds
importances = []
    
for trind, teind in stratified_16folds:
  tr = X_train[trind] 
  trl = y_train[trind]
  te = X_train[teind]
  tel = y_train[teind]

mlp = MLPClassifier(max_iter=500)

parameter_space = {
    'activation': ['tanh', 'relu','LeakyRelu'],
    'alpha': [0.0001, 0.05],
    'batch_size': [32, 64, 128],
    'early_stopping': [True],
    'hidden_layer_sizes': [(100,10), (50,50,50), (50,100,50), (100,20,20)],
    'learning_rate': ['adaptive'],
    'solver': ['sgd', 'adam'],
    'tol': [1e-4, 1e-3, 1e-2],  # Add tolerance parameter
    }
# Run the search

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, cv=10, n_jobs=-1)
clf.fit(X_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)
print(clf.best_params_)
best_estimator = clf.best_estimator_

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , clf.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

#Fitting 16 folds for each of 93312 candidates, totalling 1492992 fits
#[CV] END activation=tanh, alpha=0.0001, batch_size=32, early_stopping=True, hidden_layer_sizes=(100, 10), 
#learning_rate=constant, max_iter=100, n_iter_no_change=100, solver=sgd, tol=0.0001; total time=   7.5s
### its impossible to run this whole parameters""
