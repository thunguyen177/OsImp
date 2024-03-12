import time
import numpy as np
import random


def generate_nan(Xtrain, missing_rate):
  # this function generate NaN on a dataset that has missing values
  # the missing_rate is the ratio between the number of missing entries to be generated compared to the number of observed entries
  Xshape = Xtrain.shape
  n_missing=np.count_nonzero(np.isnan(Xtrain))
  Xtr_nan = Xtrain.flatten()
  Xtr_nan_id = list(range(Xtrain.size)) #  all index of elements after flatten data
  nan_input_id =  np.where((np.isnan(Xtr_nan.flatten().tolist() )))[0].tolist() # Find indices of NaN value (input data contains NaN)
  
  choice_list = [e for e in Xtr_nan_id if e not in nan_input_id] # Just choose not NaN elements for generating NaN
  na_id = random.sample(choice_list,round(missing_rate*(Xtrain.size - n_missing))) 
  Xtr_nan[na_id] = np.nan 
  return Xtr_nan.reshape(Xshape)

