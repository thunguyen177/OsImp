import time
import numpy as np
from functions.utils import *
from functions.dpers import *
from imblearn.metrics import geometric_mean_score, sensitivity_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn import metrics
import pandas as pd
import time

# import classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def OsImp_star(X0, y0, n, missing_rate, G, imputer):
    # n: a vector that specify how many extra samples should be generated for each class
    # non positive entries of n indicates that a class should NOT be oversampled
    # get the positive index of the class should be oversampled
    posId = np.where(n>0)[0]

    n = np.abs(n)
    idz = np.array([np.random.choice(range(sum(y0==g)),n[g]) for g in range(G)], dtype = object)
    Zover = [X0[y0==g][idz[g]] for g in posId]
    Zover = np.vstack((Zover))
    Zover = generate_nan(Zover, missing_rate) 

#     stack the extra sample with the original data
    Xnew = np.vstack((Zover, X0))
    Xnew = imputer(Xnew)    
    
    ynew = np.hstack((np.array([np.repeat(g,n[g]) for g in posId]).flatten(), #extra generated samples
                      y0))
    return Xnew, ynew

def OsImp_res_star(Xtrain, ytrain, Xtest, ytest, n, K, G, imputer, missing_rate = 0.3, tune_missing_rate = False):
  SVMclf = SVC(gamma='auto')
  if tune_missing_rate:
        best_missing_rate = opt_missing_rate(Xtrain, ytrain, n, K, SVMclf, G = G, imputer = imputer)
  else: 
        best_missing_rate = missing_rate     
    
  Xnew, ynew = OsImp_star(Xtrain, ytrain, n, missing_rate = best_missing_rate, G = G, imputer = imputer)
  Xnew = imputer(Xnew)  
  SVMclf.fit(Xnew, ynew)

  LRclf = LogisticRegression(random_state=1)
  if tune_missing_rate:
        best_missing_rate = opt_missing_rate(Xtrain, ytrain, n, K, LRclf, G = G, imputer = imputer)
  else: 
        best_missing_rate = missing_rate       
  Xnew, ynew = OsImp_star(Xtrain, ytrain, n, missing_rate = best_missing_rate, G = G, imputer = imputer)
  Xnew = imputer(Xnew)  
  LRclf.fit(Xnew,ynew)

  dctClf = DecisionTreeClassifier(random_state=0) 
  if tune_missing_rate:
        best_missing_rate = opt_missing_rate(Xtrain, ytrain, n, K, dctClf, G = G, imputer = imputer)
  else: 
        best_missing_rate = missing_rate 
  Xnew, ynew = OsImp_star(Xtrain, ytrain, n, missing_rate = best_missing_rate, G = G, imputer = imputer)
  Xnew = imputer(Xnew)
  dctClf.fit(Xnew,ynew) 
    
  res = np.array([[f1_score(ytest, SVMclf.predict(Xtest), average = 'weighted'),
                    sensitivity_score(ytest, SVMclf.predict(Xtest), average='micro')], 
                  [f1_score(ytest, LRclf.predict(Xtest), average = 'weighted'),
                   sensitivity_score(ytest, LRclf.predict(Xtest), average='micro')],
                  [f1_score(ytest, dctClf.predict(Xtest), average = 'weighted'),
                   sensitivity_score(ytest, dctClf.predict(Xtest), average='micro')]])
  return res    