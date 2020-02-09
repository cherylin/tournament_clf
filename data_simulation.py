#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:49:36 2020

@author: linqian
"""

import numpy as np
import mlpy
from sklearn import linear_model, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# generate different dataset and get average on these

# can further write it to config.json and loop for different configuration
# parameters
num_obs = 100; # number of observations
num_predictors = 4; # number of predictors
prob = 0.5 # probability that yi = 1
beta = np.random.uniform(0, 1, size=(1, num_predictors))# signal of each predictor
# multivariate normal distribution
sigma = np.eye(num_predictors) # covariance matrix
mu = np.zeros(num_predictors) # ith row for mu
# parameters for lasso model
a = 0.1 # for regularization

# may play around different generation process
# (1) generate Y dim = n*1
y = np.random.choice(a=[-1, 1], size=(num_obs,1), replace=True, p=[1-prob, prob])
# (2) generate Xi
# generate error terms dim = n*p 
U = np.random.multivariate_normal(mean=mu, cov=sigma, size=num_obs)
# multiply yi's and beta and add with U to get X, dimX = n*p
X = np.dot(y, beta) + U

# (3) Apply to classifiers and do cross-validation to assess the average accuracy
# 1: Lasso: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
clf1 = linear_model.Lasso(alpha=a)
da = mlpy.DLDA(delta=0)
def cross_validation(calssifier, X, y, k, mode="Lasso"):
    acc_test = []
    acc_train = []
    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred_test = []
        y_pred_train = []
        if mode == "Lasso":
            calssifier.fit(X_train, y_train)
            y_pred_test = calssifier.predict(X_test)
            y_pred_train = calssifier.predict(X_train)
        elif mode == "DLDA":
            calssifier.learn(X_train, np.transpose(y_train)[0])
            y_pred_test = calssifier.pred(X_test)
            y_pred_train = calssifier.pred(X_train)
        y_pred_test[y_pred_test > 0] = 1
        y_pred_test[y_pred_test <= 0] = -1
        y_pred_train[y_pred_train > 0] = 1
        y_pred_train[y_pred_train <= 0] = -1
        acc_test.append(metrics.accuracy_score(y_test, y_pred_test))
        acc_train.append(metrics.accuracy_score(y_train, y_pred_train))
    return np.array(acc_test).mean(), np.array(acc_train).mean()

acc_test, acc_train = cross_validation(clf1, X, y, 5, "Lasso")
print("LASSO: The training accuracy = {}; accuracy for test data = {}".format(acc_train, acc_test))

acc_test, acc_train = cross_validation(da, X, y, 5, "DLDA")
print("DLDA: The training accuracy = {}; accuracy for test data = {}".format(acc_train, acc_test))

