# Set work directory
import os
os.chdir("/Users/Zoey/Library/CloudStorage/OneDrive-UniversityofFlorida/PhD_Coding/26_MLP_CV16_opt")
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
swl = np.ones((len(sw),))
swf = sw[:,1:]
nsl = np.zeros((len(ns),))
nsf = ns[:,1:]

zdata = np.concatenate((swf,nsf))
est_labels = np.zeros((len(zdata),))
zlabel = np.concatenate((swl,nsl))
 
# inset train_test split
    X = zdata
    y = zlabel
# Splitting the datafram into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#printing out train and test sets
print(X_train.shape)
print(X_test.shape)
# save the data
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

####
# using the train test split function to generate training set and test set
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import itertools
import warnings
from sklearn import manifold
from sklearn.metrics import matthews_corrcoef

# Ignore warnings
warnings.simplefilter("ignore")

# Define initial method and variable    
def testv1(method = 'mlp'):
    accm = []
    senm = []
    spsm = []
    roc_aucm = []
    npvm = []
    prsm = []
    f1scorem = []
    tprs = []
    mccm = []
    mean_fpr = np.linspace(0, 1, 100)

# cross validation and apply multiple machine learning techniques
    cv = StratifiedKFold(n_splits = 16, random_state = 42, shuffle = True) # k-fold cross validation
    stratified_16folds = cv.split(X_train, y_train) #Fifteen-16 folds
    importances = []
    
    for trind, teind in stratified_16folds:
        tr = X_train[trind] 
        trl = y_train[trind]
        te = X_train[teind]
        tel = y_train[teind]
    
        if method == 'mlp':
           model = MLPClassifier(activation = 'tanh', alpha = 0.0001, batch_size = 128, solver = 'adam', hidden_layer_sizes = (100,20,20), learning_rate = 'adaptive', tol = 0.001).fit(tr,trl)
           importances.append(np.max(model.coefs_[0],axis = 1))
        
        pred = model.predict_proba(te)[:,1]
        thr = find_thr(model.predict_proba(tr)[:,1],trl)
        prr = np.where(pred > thr, 1, 0)
        est_labels[teind] = prr
        
        acc,sen,sps,roc_auc,prs,npv,f1score,mcc = performance_calculation(tel,prr,pred)
        accm.append(100*acc)
        senm.append(100*sen)
        spsm.append(100*sps)
        roc_aucm.append(100*roc_auc)
        prsm.append(100*prs)
        npvm.append(100*npv)
        f1scorem.append(100*f1score)
        mccm.append(100*mcc)
        
        fpr, tpr, thresholds = roc_curve(tel, pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    
    indices = np.argsort(abs(np.mean(importances, axis = 0)))[::-1]
    
    # Export importance values
    imp_score = pd.DataFrame(data = np.mean(importances, axis = 0), index = feat, columns = ['mean'])
    abs_mean = abs(np.mean(importances, axis = 0))
    imp_score.insert(1, 'abs(mean)', abs_mean)
    imp_score.to_csv(method + '_importance.csv', sep = ',', header = ['mean', 'abs(mean)'], index = 1, index_label = ['feature'])
    
    # Rank features by absolute importance
    sorted_features = []
    for f in range(len(indices)):
        sorted_features.append(feat[indices[f]])
    # Save confusion matrix
    plot_confusion_matrix(confusion_matrix(zlabel,est_labels),['Non-sweet','Sweet'],m + '_confusion_matrix.eps')
    
    # Save ROC curve
    plt.figure()
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucm)
    plt.plot(mean_fpr, mean_tpr, color = 'b', label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw = 2, alpha = .8)
    plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = .2, label = r'$\pm$ 1 std. dev.')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    
    plt.savefig(m + '_roc_curve.eps', format = 'eps', dpi = 1000)
    plt.close()
    return sorted_features, accm, senm*100, spsm*100, roc_aucm*100, prsm*100, npvm*100, f1scorem*100, est_labels, mccm, model
    
    fig = plt.figure(figsize = (6, 6))
    ax = plt.subplot(111)
    
    target_ids = range(2)
    tl = labels
    colors = ['darkgreen','orange']
    for i, c1, label in zip(target_ids, colors, ['Non-sweet', 'Sweet']):
        plt.scatter(rdata[tl ==  i, 0], rdata[tl ==  i, 1], c = c1, label = label)
            
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
    lgd =  ax.legend(loc = 'upper center', bbox_to_anchor = (1.2, 1), shadow = True, ncol = 1)
    plt.close(fig)

# Calculate performance
def performance_calculation(array1,array2,array3):
     tn, fp, fn, tp = confusion_matrix(array1,array2).ravel()
     total = tn+fp+fn+tp
     acc =  (tn+tp)/total
     sen = tp/(tp+fn)
     sps = tn/(tn+fp)
     prs = tp/(tp+fp)
     npv = tn/(tn+fn)
     f1score = 2* (sen*prs)/(prs+sen)
     fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
     roc_auc = metrics.auc(fpr, tpr)
     mcc =  matthews_corrcoef(array1,array2)
     
     return acc,sen,sps,roc_auc,prs,npv,f1score,mcc

def find_thr(pred,label):
    
    # Find the best threshold where FPR and FNR points cross
    minn = 100000
    thrr = 0.4
    
    for thr in np.arange(0.1,1,0.05):
        prr = np.where(pred > thr, 1, 0)
        tn, fp, fn, tp = confusion_matrix(label,prr).ravel()
        if tp+fn > 0:
            frr = fn/(tp+fn)
        else:
            frr = 0
        if tn+fp > 0:    
            far = fp/(tn+fp)
        else:
            far = 0 
        if np.abs(frr - far) < minn:
            minn = np.abs(frr - far)
            thrr = thr
            
    return thrr
 
def plot_confusion_matrix(cm, classes, name, normalize = False, cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(name, format = 'eps', dpi = 1000)
    plt.close()
    
# Test models
methods = ['mlp']
file = open('performance.txt',"w")
file.write('Method, Accuracy, Sensitivity, Specificity, AUC, Precision, Negative Predictive Value, F1 Score, MCC \n')

for m in methods:
    sorted_features, accm, senm, spsm, roc_aucm, prsm, npvm, f1scorem, est_labels, mccm, model = testv1(m)
    file.write(m + ':, ')
    file.write("%0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f,  %0.2f \u00B1 %0.2f \n" % (np.mean(accm),np.std(accm),np.mean(senm),np.std(senm),np.mean(spsm),np.std(spsm),np.mean(roc_aucm),np.std(roc_aucm),np.mean(prsm),np.std(prsm),np.mean(npvm),np.std(npvm),np.mean(f1scorem),np.std(f1scorem),np.mean(mccm),np.std(mccm)))
    df = pd.DataFrame(data = {"features": sorted_features})
    df.to_csv(m +'_ranked_features.csv', sep = ',',index = False)
    np.savetxt(m+'_estimated_labels.csv', est_labels.astype(np.int),delimiter = ', ',fmt = '%d')
    
    exec ("mdl_%s=model"%m[-1])
    if m == 'rf' or m == 'gbt':
      exec ("importance_%s = model.feature_importances_"%m[-1])
    if m == 'lr-l1' or m == 'lr-l2' or m == 'svm':
      exec ("coef_%s = model.coef_"%m[-1])
      exec ("intercept_%s = model.intercept_"%m[-1])
    if m == 'mlp':
      exec ("coef_%s=model.coefs_"%m[-1])
      exec ("intercepts_%s = model.intercepts_"%m[-1])

file.close()

## ----------------------------External test----------------------------
#######--------------------MLP model on test sets--------------########
y_pred_mlp = mdl_p.predict(X_test)
y_3=y_pred_mlp

# initialize counters
tp = tn = fp = fn = 0

# calculate TP, TN, FP, FN
for i in range(len(y_3)):
  if y_3[i] == 1 and y_test[i] == 1:
    tp += 1
  elif y_3[i] == 0 and y_test[i] == 0:
    tn += 1
  elif y_3[i] == 1 and y_test[i] == 0:
    fp += 1
  elif y_3[i] == 0 and y_test[i] == 1:
    fn += 1

# print the results
print("True Positives (TP):", tp)
print("True Negaitives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)

# calculate evaluation matrics
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,matthews_corrcoef,roc_auc_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_3)
sensitivity = recall_score(y_test, y_3)
specificity = tn / (tn + fp)
precision = precision_score(y_test, y_3)
f1 = f1_score(y_test, y_3)
mcc = matthews_corrcoef(y_test, y_3)
auc = roc_auc_score(y_test, y_3)

# Calculate standard deviation for evaluation metrics
metrics = [accuracy, sensitivity, specificity, precision, f1, mcc, auc]
std_dev = np.std(metrics)

# Print the results including standard deviation
print("Accuracy:", accuracy)
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1-Score:", f1)
print("MCC (Matthews Correlation Coefficient):", mcc)
print("AUC (Area Under the ROC Curve):", auc)
print("Standard Deviation of Metrics:", std_dev)

##---------choose MLP to predict data from molecular network-----------######
# Import test data
td = pd.read_csv('TestD_FS_Zscaling.csv')
tdv = td.iloc[0:,1:]
tdv

# Predict label
pred_mlp = mdl_p.predict(tdv)
print(pred_mlp)
pred_proba_mlp = mdl_p.predict_proba(tdv)[:, 1]
print(pred_proba_mlp)

print(X_train.shape)

