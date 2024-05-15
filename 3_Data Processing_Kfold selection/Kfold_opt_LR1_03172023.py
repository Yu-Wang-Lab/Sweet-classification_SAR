# set work directory
import os
os.chdir("/Users/Zoey/Library/CloudStorage/OneDrive-UniversityofFlorida/PhD_Coding/3_KFold")
os.getcwd()

# Codes are referenced from the website: https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/

# evaluate a logistic regression model using k-fold cross-validation
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# create dataset
# load data
sw = pd.read_csv ('SW_904_zscaled.csv', sep=',',encoding='utf-8')
ns = pd.read_csv ('NS_318_zscaled.csv', sep=',',encoding='utf-8')
sw = sw.dropna()
ns = ns.dropna()
    
    # features names
    feat = list(sw)
    feat = feat[1:] # feature name
    
    # convert it to numpy array
    sw = sw.values
    ns = ns.values
    
    # features and labels
    swl = np.zeros((len(sw),))
    swf = sw[:,1:]
    nsl = np.ones((len(ns),))
    nsf = ns[:,1:]

    # cross validation and apply LR1 machine learning techniques
    cv = KFold(n_splits = 9, random_state = 42, shuffle = True) # k-fold cross validation
    wdata = np.concatenate((swf,nsf))
    est_labels = np.zeros((len(wdata),))
    wlabel = np.concatenate((swl,nsl))
    stratified_9folds = cv.split(wdata, wlabel) #nine folds
    
for trind, teind in stratified_9folds:
        # k-1 datasets as training set           
        tr = wdata[trind] 
        trl = wlabel[trind] # training set label
        # 1 dataset as test set
        te = wdata[teind]
        tel = wlabel[teind]

# create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty = 'l1', solver = 'liblinear').fit(tr,trl)
# evaluate model
scores = cross_val_score(model, wdata, wlabel, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

## the results gives 0.786 of accuracy with std of 0.043 for 9 fold cross-validation.
## here is to optimize the best kfold for LR1 model.

from numpy import mean
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
 
# create the dataset
X = wdata
y = wlabel
 
# retrieve the model to be evaluate
def get_model():
 model = LogisticRegression(penalty = 'l1', solver = 'liblinear').fit(tr,trl)
 return model
 
# evaluate the model using a given test condition
def evaluate_model(cv):
 # get the model
 model = get_model()
 # evaluate the model
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 # return scores
 return mean(scores), scores.min(), scores.max()
 
# calculate the ideal test condition
ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)
# define folds to test
folds = range(2,31)
# record mean and min/max of each set of results
means, mins, maxs = list(),list(),list()
# evaluate each k value
for k in folds:
 # define the test condition
 cv = KFold(n_splits=k, shuffle=True, random_state=42)
 # evaluate k value
 k_mean, k_min, k_max = evaluate_model(cv)
 # report performance
 print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
 # store mean accuracy
 means.append(k_mean)
 # store min and max relative to the mean
 mins.append(k_mean - k_min)
 maxs.append(k_max - k_mean)
# line plot of k mean values with min/max error bars
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# plot the ideal case in a separate color
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
# show the plot
pyplot.show()
fig = pyplot.figure(figsize = (6, 6))
pyplot.close()

#orange dots in plots

