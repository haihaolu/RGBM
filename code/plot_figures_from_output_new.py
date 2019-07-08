import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_figures(name):
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

if __name__ == "__main__":
  names = ["a1a"]
  for name in names:
    plot_figures(name)