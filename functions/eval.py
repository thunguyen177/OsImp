import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.datasets import fetch_datasets

from imblearn.over_sampling import KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from functions.OsImp import *
from functions.OsImp_star import *

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


def get_res_fr_method(Xtrain,Xtest,ytrain,ytest, imba_method):
    
  if imba_method is None:
    return []
    
  Xnew,ynew = imba_method.fit_resample(Xtrain, ytrain) 
  SVMclf = SVC(gamma='auto')
  SVMclf.fit(Xnew, ynew)
  LRclf = LogisticRegression(random_state=1)
  LRclf.fit(Xnew,ynew)
  dctClf = DecisionTreeClassifier(random_state=0)  
  dctClf.fit(Xnew,ynew)  

  res = np.array([[f1_score(ytest, SVMclf.predict(Xtest), average = 'weighted'),
                    sensitivity_score(ytest, SVMclf.predict(Xtest), average='micro')], 
                  [f1_score(ytest, LRclf.predict(Xtest), average = 'weighted'),
                   sensitivity_score(ytest, LRclf.predict(Xtest), average='micro')],
                  [f1_score(ytest, dctClf.predict(Xtest), average = 'weighted'),
                   sensitivity_score(ytest, dctClf.predict(Xtest), average='micro')]])    
  return res  

def get_all_res_star(X, y, i, n, K, G, imputer, ori_missing_rate):
  # ori_missing_rate: the simulated missing rate on the original data 
    
  Xtrain, Xtest, ytrain, ytest = train_test_split(X,y.reshape((-1,1)),test_size=0.4)
  ytrain = ytrain.flatten() 

  #introduce missingness into original data
  Xtrain =  generate_nan(Xtrain, ori_missing_rate)    

  scaler = StandardScaler()
  scaler.fit(Xtrain)
  Xtrain = scaler.transform(Xtrain)
  Xtest = scaler.transform(Xtest)

  #impute missing data for other approaches
  Xtrain_imp = imputer(Xtrain) 

  # build a dictionary of number samples to use for samply_strategy for methods in imbalance-learn package 
  n_per_class = np.array([sum(ytrain == g) for g in range(G)])  
  keys = np.arange(G)
  values = n
  values[values<0] =0
  n_dict = dict(zip(keys, values + n_per_class)) 
              
        
#   res = np.hstack([ADASYN_result, 
  res = np.hstack([get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, NearMiss()), 
                   get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, 
                                     KMeansSMOTE(sampling_strategy = n_dict, random_state=42,cluster_balance_threshold=0)),
                get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, SMOTEENN(sampling_strategy = n_dict, random_state=42)),
                get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, SVMSMOTE(sampling_strategy = n_dict, random_state=42)),
                get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, InstanceHardnessThreshold(random_state=0)),
                OsImp_res_star(Xtrain, ytrain,Xtest, ytest, n, K, G, imputer)])
                     
  return res

def get_all_res(X, y, i, n, K, G, imputer, n_dict):
  Xtrain, Xtest, ytrain, ytest = train_test_split(X,y.reshape((-1,1)),test_size=0.4)
  ytrain = ytrain.flatten() 

  scaler = StandardScaler()
  scaler.fit(Xtrain)
  Xtrain = scaler.transform(Xtrain)
  Xtest = scaler.transform(Xtest)
 
        
  res = np.hstack([get_res_fr_method(Xtrain_imp,Xtest,ytrain,ytest, NearMiss()),  
                   get_res_fr_method(Xtrain,Xtest,ytrain,ytest, KMeansSMOTE(sampling_strategy = n_dict,
                                                                            random_state=42,cluster_balance_threshold=0)),
                get_res_fr_method(Xtrain,Xtest,ytrain,ytest, SMOTEENN(sampling_strategy = n_dict,random_state=42)),
                get_res_fr_method(Xtrain,Xtest,ytrain,ytest, SVMSMOTE(sampling_strategy = n_dict,random_state=42)),
                get_res_fr_method(Xtrain,Xtest,ytrain,ytest, InstanceHardnessThreshold(sampling_strategy = n_dict,random_state=0)),
                OsImp_res(Xtrain, ytrain,Xtest, ytest, n, K, G, imputer)])
                     
  return res


def show_result(res):
    
  # return result that contains the average of all runs, and result_latex contains the results for latex table construction & display
  result = []
  res_mean = np.mean(res, axis = 0)
  f1_res = res_mean[:, 2*np.arange(6)]
  sensi_res  = res_mean[:, 2*np.arange(6)+1]

  f1_res = pd.DataFrame(f1_res,
                         columns = ['NearMiss','kmeanSmt','SmtNN','SVMSmt','InsHard','OsImp'],
                         index = ['SVM','LR','DCT']).astype(float).round(3)  

  sensi_res = pd.DataFrame(sensi_res,
                         columns = ['NearMiss','kmeanSmt','SmtNN','SVMSmt','InsHard', 'OsImp'],
                         index = ['SVM','LR','DCT']).astype(float).round(3) 
  result.append((f1_res,sensi_res)) 


  res_std = np.std(res.astype(float), axis = 0)
  f1_res = res_std[:, 2*np.arange(6)]
  sensi_res  = res_std[:, 2*np.arange(6)+1]

  f1_res = pd.DataFrame(f1_res,
                         columns = ['NearMiss','kmeanSmt','SmtNN','SVMSmt','InsHard', 'OsImp'],
                         index = ['SVM','LR','DCT']).astype(float).round(3)   

  sensi_res = pd.DataFrame(sensi_res,
                         columns = ['NearMiss','kmeanSmt','SmtNN','SVMSmt','InsHard', 'OsImp'],
                         index = ['SVM','LR','DCT']).astype(float).round(3)  
  result.append((f1_res,sensi_res)) 



  # build table for displaying the results and getting the latex codes 
  pm = pd.DataFrame(np.repeat("$\pm$", f1_res.size).reshape(f1_res.shape),
                 index = f1_res.index,
                 columns = f1_res.columns)    
  result_latex = []
  result_latex.append(result[0][0].astype(str) +pm+ result[1][0].astype(str))
  result_latex.append(result[0][1].astype(str) +pm+ result[1][1].astype(str))  
    
  return result, result_latex    