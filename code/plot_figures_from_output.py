import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_figures(num_iter, name):

  def set_up_figure(if_train):
    plt.legend(prop={'size': 12})
    if if_train:
      if name == "a9a":
        plt.xlim(0, 50)
        plt.ylim(0.001, 1)
      elif name == "colon-cancer":
        plt.xlim(0, 50)
        plt.ylim(0.00001, 1)
      elif name == "rcv1":
        plt.xlim(0, 1000)
        plt.ylim(0.01,1)
      elif name == "YearPredictionMSD_t":
        plt.xlim(0, 200)
        plt.ylim(0.1,100)
      elif name == "duke":
        plt.xlim(0, 100)
        plt.ylim(0.000001,1)
    else:
      if name == "a9a":
        plt.xlim(0, 50)
      elif name == "colon-cancer":
        plt.xlim(0, 50)
      elif name == "rcv1":
        plt.xlim(0, 1000)
      elif name == "YearPredictionMSD_t":
        plt.xlim(0, 200)
        plt.ylim(30,60)
      elif name == "duke":
        plt.xlim(0, 100)

  matfile = "../output/output_" + name + "_iter_" + str(num_iter) + ".mat"
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
    fig = plt.figure(figsize=(8, 6.5))
    xlims = np.zeros(5)
    for i in range(5):
      xlims[i] = sample_size[i]*len(losses[j][str(i)])/sample_size[0]
      if j == 0:
        plt.semilogy(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)]-train_min, color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        # plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(True)
      else:
        plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_count')

  for j in range(2):
    fig = plt.figure(figsize=(8, 6.5))
    xlims = np.zeros(5)
    for i in range(5):
      xlims[i] = running_time[i]
      if j == 0:
        # plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        plt.semilogy(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)]-train_min, color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(True)
      else:
        plt.plot(np.linspace(0, xlims[i], len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(False)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_time')

  for j in range(2):
    fig = plt.figure(figsize=(8, 6.5))
    for i in range(5):
      if j == 0:
        plt.semilogy(np.arange(0, len(losses[j][str(i)])), losses[j][str(i)]-train_min, color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        # plt.plot(np.arange(0, len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(True)
        if name == "rcv1":
          plt.xlim(0, 500)
      else:
        plt.plot(np.arange(0, len(losses[j][str(i)])), losses[j][str(i)], color=colors[i], linestyle=linestyles[i], linewidth=2.0, label='t='+str(sample_size[i]))
        set_up_figure(False)
        if name == "rcv1":
          plt.xlim(0, 500)
    fig.savefig('../figures/'+name+'_'+losses_name[j]+'_iteration')

if __name__ == "__main__":
  names = ["a9a", "rcv1", "YearPredictionMSD_t", "colon-cancer", "duke"]
  num_iter = [100, 10, 100, 100, 100]
  for i in range(len(names)):
    plot_figures(num_iter[i], names[i])