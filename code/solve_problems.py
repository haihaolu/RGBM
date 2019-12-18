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

def save_output(gbts_list, num_iter, name):
  """
  Saves the output of RGBM.
  """
  matfile = "../output/output_" + name + "_iter_" + str(num_iter)
  scipy.io.savemat(matfile, 
    mdict={'train_loss_0': np.array(gbts_list[0][0].train_loss),
           'train_loss_1': np.array(gbts_list[1][0].train_loss),
           'train_loss_2': np.array(gbts_list[2][0].train_loss), 
           'train_loss_3': np.array(gbts_list[3][0].train_loss),
           'train_loss_4': np.array(gbts_list[3][0].train_loss),  
           'test_loss_0': np.array(gbts_list[0][0].test_loss),
           'test_loss_1': np.array(gbts_list[1][0].test_loss),
           'test_loss_2': np.array(gbts_list[2][0].test_loss), 
           'test_loss_3': np.array(gbts_list[3][0].test_loss),
           'test_loss_4': np.array(gbts_list[4][0].test_loss),
           'sample_size': np.array([gbts_list[i][1] for i in range(len(gbts_list))]), 
           'running_time': np.array([gbts_list[i][2] for i in range(len(gbts_list))])})

def plot_figures(num_iter, name):
  """
  Plots the figures.
  """
  
  def set_up_figure(if_log):
    plt.legend(prop={'size': 12})
    if name == "a9a":
      plt.xlim(0, 100)
    elif name == "colon-cancer":
      plt.xlim(0, 100)
    elif name == "rcv1":
      plt.xlim(0, 50)
    elif name == "YearPredictionMSD":
      plt.xlim(0, 100)
      if not if_log:
        plt.ylim(40,60)

  matfile = "output_" + name + "_iter_" + str(num_iter) + ".mat"
  data = scipy.io.loadmat("../output/" + matfile)

  train_loss = {"0": data['train_loss_0'][0], "1": data['train_loss_1'][0], "2": data['train_loss_2'][0], "3": data['train_loss_3'][0], "4": data['train_loss_4'][0]}

  test_loss = {"0": data['test_loss_0'][0], "1": data['test_loss_1'][0], "2": data['test_loss_2'][0], "3": data['test_loss_3'][0], "4": data['test_loss_4'][0]}
  sample_size, running_time = data["sample_size"][0], data["running_time"][0]
  train_min = np.min(np.concatenate([train_loss[str(i)] for i in range(4)]))

  losses = [train_loss, test_loss]
  losses_name = ["train_loss", "test_loss"]
  colors = ['blue', 'orange', 'green', 'black', 'red']
  # linestyles = [':', '-.', '--', '-', '-']
  linestyles = ['-', '-', '-', '-', '-']
  # makers = ['.', 'o.', 'v', '<', '>']

  plt.rc('font', family='serif')
  plt.rc('xtick',labelsize=15)
  plt.rc('ytick',labelsize=15)

  for j in range(2):
    fig = plt.figure()
    xlims = np.zeros(5)
    for i in range(5):
      xlims[i] = sample_size[i]*len(losses[j][str(i)])/sample_size[0]
      plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
    plt.xlim(0, min(xlims))
    set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_count')

  for j in range(2):
    fig = plt.figure()
    xlims = np.zeros(5)
    for i in range(5):
      xlims[i] = running_time[i]
      plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
    plt.xlim(0, min(xlims))
    set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_time')

  for j in range(2):
    fig = plt.figure()
    for i in range(5):
      plt.plot(np.arange(0, len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
    plt.xlim(0, len(train_loss[str(0)]-1))
    set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_iteration')

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

    ori_iter = int(sys.argv[1])
    num_iter = int(sys.argv[1])

    for sample_size in np.logspace(np.log10(p), np.log10(1), num=5, dtype=int):

      gbts = GBTS(loss, derivative, sample_size, step_size)
      start_time = time.time()
      gbts.train(train_data, num_iter, test_set=test_data)
      running_time = time.time() - start_time
      gbts_list.append((gbts, sample_size, running_time))
      num_iter = int(num_iter * np.exp(np.log(p)/4))
      print("t={}: final training loss={}, final testing loss={}.".format(
        sample_size, gbts.train_loss[-1], gbts.test_loss[-1]))
      del gbts

    save_output(gbts_list, ori_iter, name)
    plot_figures(ori_iter, name)
    print("Finish training {}.".format(name))
