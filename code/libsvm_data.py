from __future__ import division
from rgbm import Dataset, GBTS, TreeStump
from sklearn.metrics import mean_squared_error
import numpy as np
import cProfile
from rgbm import Dataset, GBTS
import urllib
import scipy.io
import sklearn.datasets
from sklearn.model_selection import train_test_split
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functional as F
import time

def SetupData(name):
  """
  Loads the dataset and randomly splits the dataset into 80% training dataset and 20% testing dataset.
  """
  if name in ["duke", "colon-cancer", "YearPredictionMSD_t", "YearPredictionMSD"]:
    data = sklearn.datasets.load_svmlight_file("../dataset/" + name + ".bz2")
  elif name == "rcv1":
    data = sklearn.datasets.load_svmlight_file("../dataset/rcv1_train.binary")
  else:
    data = sklearn.datasets.load_svmlight_file("../dataset/" + name + ".txt")
  X = np.asarray(data[0].todense())
  y = np.array(data[1])

  return train_test_split(X, y, test_size=0.2, random_state=40)

def save_output(gbts_list, name):
  """
  Saves the output of RGBM.
  """
  matfile = "../output/output_" + name
  scipy.io.savemat(matfile, 
    mdict={'train_loss_0': np.array(gbts_list[0][0].train_loss),
           'train_loss_1': np.array(gbts_list[1][0].train_loss),
           'train_loss_2': np.array(gbts_list[2][0].train_loss), 
           'train_loss_3': np.array(gbts_list[3][0].train_loss), 
           'test_loss_0': np.array(gbts_list[0][0].test_loss),
           'test_loss_1': np.array(gbts_list[1][0].test_loss),
           'test_loss_2': np.array(gbts_list[2][0].test_loss), 
           'test_loss_3': np.array(gbts_list[3][0].test_loss), 
           'sample_size': np.array([gbts_list[i][1] for i in range(len(gbts_list))]), 
           'running_time': np.array([gbts_list[i][2] for i in range(len(gbts_list))])})


def plot_figures(name):
  """
  Plots the training and testing performance of the algorithm.
  """
  matfile = "output_" + name + ".mat"
  data = scipy.io.loadmat("../output/" + matfile)

  train_loss = {"0": data['train_loss_0'][0], "1": data['train_loss_1'][0], "2": data['train_loss_2'][0], "3": data['train_loss_3'][0]}
  test_loss = {"0": data['test_loss_0'][0], "1": data['test_loss_1'][0], "2": data['test_loss_2'][0], "3": data['test_loss_3'][0]}
  sample_size, running_time = data["sample_size"][0], data["running_time"][0]

  fig = plt.figure()
  xlims = np.zeros(4)
  for i in range(4):
    xlims[i] = sample_size[i]*len(train_loss[str(i)])
    plt.plot(np.linspace(1, xlims[i], len(train_loss[str(i)])), train_loss[str(i)], label='t='+str(sample_size[i]))
  plt.xlim(1, min(xlims))
  plt.xscale('log')
  if name == "YearPredictionMSD_t":
    plt.yscale('log')
  plt.legend()
  fig.savefig('../figures/'+name+'_train_loss_count')

  fig = plt.figure()
  for i in range(4):
    plt.plot(np.arange(1, 1+len(train_loss[str(i)])), train_loss[str(i)], label='t='+str(sample_size[i]))
  plt.xlim(1, len(train_loss[str(0)]))
  plt.xscale('log')
  if name == "YearPredictionMSD_t":
    plt.yscale('log')
  plt.legend()
  fig.savefig('../figures/'+name+'_train_loss_iteration')

  fig = plt.figure()
  xlims = np.zeros(4)
  for i in range(4):
    xlims[i] = sample_size[i]*len(test_loss[str(i)])
    plt.plot(np.linspace(1, xlims[i], len(test_loss[str(i)])), test_loss[str(i)], label='t='+str(sample_size[i]))
  plt.xlim(1, min(xlims))
  plt.xscale('log')
  if name == "YearPredictionMSD_t":
    plt.yscale('log')
  plt.legend()
  fig.savefig('../figures/'+name+'_test_loss_count')

  fig = plt.figure()
  for i in range(4):
    plt.plot(np.arange(1, 1+len(train_loss[str(i)])), test_loss[str(i)], label='t='+str(sample_size[i]))
  plt.xlim(1, 1+len(test_loss[str(0)]))
  plt.xscale('log')
  if name == "YearPredictionMSD_t":
    plt.yscale('log')
  plt.legend()
  fig.savefig('../figures/'+name+'_test_loss_iteration')


"""
Trains the RGBM model, stores the outputs of the algorithm and compares the results.
"""
if __name__ == "__main__":
  
  if len(sys.argv) == 2:
    names = ["a9a", "YearPredictionMSD_t", "colon-cancer", "rcv1"]
  else:
    names = sys.argv[2:]
  for name in names:
    X_train, X_test, y_train, y_test = SetupData(name)
    print(len(y_train), len(y_test))
    train_data = Dataset(X_train, y_train)
    test_data = Dataset(X_test, y_test)

    if name in ["YearPredictionMSD_t", "housing", "space_ga", "triazines", "YearPredictionMSD"]:
      loss = F.l2
      derivative = F.d_l2
      step_size = 1.0
    else:
      loss = F.r_logistic
      derivative = F.d_r_logistic
      step_size = 4.0

    gbts_list = []
    p = np.shape(X_train)[1]

    num_iter = int(sys.argv[1])

    for sample_size in np.logspace(np.log10(p), np.log10(1), num=4, dtype=int):

      gbts = GBTS(loss, derivative, sample_size, step_size)
      start_time = time.time()
      gbts.train(train_data, num_iter, test_set=test_data)
      running_time = time.time() - start_time
      gbts_list.append((gbts, sample_size, running_time))
      num_iter = int(num_iter * np.exp(np.log(p)/3))
      print("sample_size:" + str(sample_size) + ": running time: " + str(running_time))

    save_output(gbts_list, name)
    plot_figures(name)
    print("Finish training " + name)
  




