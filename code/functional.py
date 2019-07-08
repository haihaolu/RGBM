from __future__ import division
import numpy as np

def logistic(predict,label):
  return np.log(1+np.exp(-predict*label))

def d_logistic(predict,label):
  temp = np.exp(-label*predict)
  return label*temp/(1+temp)

def r_logistic(predict,label):
  return np.log(1+np.exp(-predict*label)) + 0.0001/2*(predict-label)**2

def d_r_logistic(predict,label):
  temp = np.exp(-label*predict)
  return label*temp/(1+temp) + 0.0001*(label-predict)

def l2(predict, label):
  return 1/2*(predict-label)**2

def d_l2(predict, label):
  return label-predict