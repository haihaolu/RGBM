import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_figures(name):
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

  matfile = "output_" + name + ".mat"
  data = scipy.io.loadmat("../output/" + matfile)

  train_loss = {"0": data['train_loss_0'][0], "1": data['train_loss_1'][0], "2": data['train_loss_2'][0], "3": data['train_loss_3'][0]}

  test_loss = {"0": data['test_loss_0'][0], "1": data['test_loss_1'][0], "2": data['test_loss_2'][0], "3": data['test_loss_3'][0]}
  sample_size, running_time = data["sample_size"][0], data["running_time"][0]
  train_min = np.min(np.concatenate([train_loss[str(i)] for i in range(4)]))

  losses = [train_loss, test_loss]
  losses_name = ["train_loss", "test_loss"]
  colors = ['blue', 'orange', 'green', 'red']
  linestyles = [':', '-.', '--', '-']

  plt.rc('font', family='serif')
  plt.rc('xtick',labelsize=15)
  plt.rc('ytick',labelsize=15)

  for j in range(2):
    fig = plt.figure()
    xlims = np.zeros(4)
    for i in range(4):
      xlims[i] = sample_size[i]*len(losses[j][str(i)])/sample_size[0]
      plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
    plt.xlim(0, min(xlims))
    set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_count')

  for j in range(2):
    fig = plt.figure()
    for i in range(4):
      plt.plot(np.arange(0, len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
    plt.xlim(0, len(train_loss[str(0)]))
    set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_iteration')

  fig = plt.figure()
  for i in range(4):
    plt.plot(np.linspace(0, xlims[i], len(train_loss[str(i)])), train_loss[str(i)] - train_min, color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
  plt.yscale('log')
  set_up_figure(True)
  fig.savefig('../figures/'+name+'_'+losses_name[j]+'_count_log')

if __name__ == "__main__":
  names = ["a1a"]
  for name in names:
    plot_figures(name)