from __future__ import division
import sys
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import time

class Dataset(object):
  """
  The Dataset, containing feature vectors and corresponding labels or continuous responses.
  """

  def __init__(self, X, y):
    self.X = X
    self.y = y

class TreeStump(object):
  """
  A tree stump.
  """

  def __init__(self, n, sample_size, if_quantile):
    self.split_feature_id = None
    self.split_val = None
    self.left_num = 0
    self.right_num = n
    self.left_weight = 0
    self.right_weight = 0
    self.coefficient = None
    self.sample_size = sample_size
    self.if_quantile = if_quantile

  def __str__(self):
    return "split_feature_id: %d\nsplit_val: %f\nleft_num: %d\nright_num: %d\nleft_weight: %f\nright_weight: %f\ncoefficient: %f" % (self.split_feature_id, self.split_val, self.left_num, self.right_num, self.left_weight, self.right_weight, self.coefficient)

  def __repr__(self):
    return "split_feature_id: %d\nsplit_val: %f\nleft_num: %d\nright_num: %d\nleft_weight: %f\nright_weight: %f\ncoefficient: %f" % (self.split_feature_id, self.split_val, self.left_num, self.right_num, self.left_weight, self.right_weight, self.coefficient)

  def _calc_split_gain(self, g_l, g_r, n_l, n_r):
    if n_l == 0:
      return g_r**2/n_r
    elif n_r == 0:
      return g_l**2/n_l
    else:
      return g_l**2/n_l + g_r**2/n_r

  def _calc_leaf_weight(self, g_l, g_r, n_l, n_r):
    if n_r == 0:
      self.left_weight = 1/n_l*np.sign(g_l)
      self.right_weight = 0
    elif n_l == 0:
      self.left_weight = 0
      self.right_weight = 1/n_r*np.sign(g_r)
    else:
      temp = 1/np.sqrt(g_l**2/n_l + g_r**2/n_r)
      self.left_weight = temp * g_l / n_l
      self.right_weight = temp * g_r / n_r

  def _calc_coeff(self, g_l, g_r):
    self.coefficient = (g_l * self.left_weight + g_r * self.right_weight)

  def _revised_quantile(self, sorted_values, quantiles):
    revised_quantiles=np.zeros(len(quantiles), dtype=int)
    i,j = 0,0
    for j in range(len(quantiles)):
      while i < len(sorted_values) and sorted_values[i] <= sorted_values[quantiles[j]]:
        i = i+1
      revised_quantiles[j] = i-1
    revised_quantiles = np.append([-1], revised_quantiles)
    revised_quantiles = np.unique(revised_quantiles)
    # print revised_quantiles
    return revised_quantiles

  def build(self, instances, grad):
    """
    Builds a tree strump given the gradient and data.
    """
    best_gain = 0.
    best_feature_id = None
    best_val = 0.
    best_g_l = None
    best_g_r = None
    best_n_l = None
    best_n_r = None

    g = np.sum(grad)
    n = instances.shape[0]
    p = instances.shape[1]
    if self.if_quantile:
      num_quantile = 100
    else:
      num_quantile = n

    id_quantile = np.linspace(0, n-1, num_quantile, dtype=int)

    if self.sample_size > p:
      raise Exception('sample_size cannot be larger than p.')

    set_of_features = np.random.choice(p, self.sample_size, replace=False)

    for feature_id in set_of_features:
      g_l = 0.
      g_r = np.sum(grad)
      sorted_values = np.sort(instances[:, feature_id])
      sorted_instance_ids = instances[:, feature_id].argsort()
      revised_quantiles = self._revised_quantile(sorted_values, id_quantile)
      num_revised_quantiles = len(revised_quantiles)

      for j in range(0, num_revised_quantiles-1):
        
        g_l += np.sum(grad[sorted_instance_ids[(1+revised_quantiles[j]):(1+revised_quantiles[j+1])]])
        g_r = g - g_l

        current_gain = self._calc_split_gain(g_l, g_r, 1+revised_quantiles[j+1], n-1-revised_quantiles[j+1])
        if current_gain > best_gain:
          best_gain = current_gain
          self.split_feature_id = feature_id
          self.split_val = instances[sorted_instance_ids[revised_quantiles[j+1]], feature_id]
          best_g_l = g_l
          best_g_r = g_r
          best_n_l = 1 + revised_quantiles[j+1]
          best_n_r = n - 1 - revised_quantiles[j+1]
          best_sorted_instance_ids = sorted_instance_ids
          best_revised_quantiles = revised_quantiles

    self.left_num = best_n_l
    self.right_num = best_n_r
    self._calc_leaf_weight(best_g_l, best_g_r, best_n_l, best_n_r)
    self._calc_coeff(best_g_l, best_g_r)
    
  def predict(self, x):
    """
    Predicts the output value for a feature vector x using the tree stump model.
    """
    if x[self.split_feature_id] <= self.split_val:
      return self.left_weight
    else:
      return self.right_weight

class GBTS(object):
  """
  Gradient boosting tree stumps.
  """
  def __init__(self, loss_function, derivative, sample_size, step_size, if_quantile=True):
    self.best_iteraiton = None
    self.loss_function = loss_function
    self.derivative = derivative
    self.models = []
    self.sample_size = sample_size
    self.if_quantile = if_quantile
    self.train_scores = None
    self.test_scores = None
    self.train_loss = np.array([])
    self.test_loss = np.array([])
    self.step_size = step_size

  def __str__(self):
    return [m for m in models]

  def _update_training_data_scores(self, train_set, models):
    X = train_set.X
    y = train_set.y
    if len(models) == 0:
      self.train_scores = np.zeros(len(y))
    else:
      for i in range(len(y)):
        learner = models[-1]
        self.train_scores[i] += self.step_size * learner.coefficient * learner.predict(X[i, :])

  def _update_testing_data_scores(self, test_set, models):
    X = test_set.X
    y = test_set.y
    if len(models) == 0:
      self.test_scores = np.zeros(len(y))
    else:
      for i in range(len(y)):
        learner = models[-1]
        self.test_scores[i] += self.step_size * learner.coefficient * learner.predict(X[i, :])
  
  def _calc_gradient(self, train_set):
    labels = train_set.y
    grad = self.derivative(self.train_scores, labels)
    return grad

  def _build_learner(self, train_set, grad):
    learner = TreeStump(len(train_set.y), self.sample_size, self.if_quantile)
    learner.build(train_set.X, grad)
    return learner

  def _compute_loss(self, scores, dataset):
    return np.mean(self.loss_function(scores, dataset.y))

  def train(self, train_set, num_boost_round=10, test_set=None):
    """
    Trains a boosted tree model.
    """
    models = []
    frequency_store_test_loss = num_boost_round // 1000
    start_time = time.time()

    for iter_count in range(num_boost_round):
      self._update_training_data_scores(train_set, models)
      if test_set is not None:
        self._update_testing_data_scores(test_set, models)

      grad = self._calc_gradient(train_set)
      learner = self._build_learner(train_set, grad)
      models.append(learner)
      self.models = models[:]
      self.train_loss = np.append(self.train_loss, self._compute_loss(self.train_scores, train_set))
      if test_set is not None:
        self.test_loss = np.append(self.test_loss, self._compute_loss(self.test_scores, test_set))
      # print(self.train_loss[-1])

    self._update_training_data_scores(train_set, models)
    self.train_loss = np.append(self.train_loss, self._compute_loss(self.train_scores, train_set))
    if test_set is not None:
        self.test_loss = np.append(self.test_loss, self._compute_loss(self.test_scores, test_set))
    
  def predict(self, x, models=None):
    """
    Predicts the output value for a feature vector x using a boosted tree stump model.
    """
    if models is None:
      models = self.models
    return np.sum(self.step_size * m.coefficient * m.predict(x) for m in models)


