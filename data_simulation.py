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

def generate_data():
    """Generate X, y based on the chosen generationg process, may play around different generation process."""
    # (1) generate Y dim = n*1
    y = np.random.choice(a=[-1, 1], size=(num_obs,1), replace=True, p=[1-prob, prob])
    # (2) generate Xi
    # generate error terms dim = n*p 
    U = np.random.multivariate_normal(mean=mu, cov=sigma, size=num_obs)
    # multiply yi's and beta and add with U to get X, dimX = n*p
    X = np.dot(y, beta) + U
    return X, y

def split_data(X, y):
    """Split the given dataset."""
    # take the first 80% data as the training data
    skf = StratifiedKFold(n_splits=5)
    # only get the first set of indices
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, y_train, X_test, y_test

def train_test_lasso(alpha, X_train, y_train, X_test, y_test):
    """Train a lasso model based on given data, and assess its training/test error."""
    # 1: Lasso: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    clf = linear_model.Lasso(alpha=a)
    clf.fit(X_train, y_train)
    # calculate training accuracy and test accuracy
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    y_pred_test[y_pred_test > 0] = 1
    y_pred_test[y_pred_test <= 0] = -1
    y_pred_train[y_pred_train > 0] = 1
    y_pred_train[y_pred_train <= 0] = -1
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    return acc_train, acc_test

def train_test_dlda(d, X_train, y_train, X_test, y_test):
    clf = mlpy.DLDA(delta=d)
    y_train = np.transpose(y_train)[0]
    clf.learn(X_train, y_train)
    y_pred_train = clf.pred(X_train)
    y_pred_test = clf.pred(X_test)
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    return acc_train, acc_test

def avg_train_test_acc(alpha, delta):
    """Given a list of classifer model, average their training/test errors on N datasets."""
    # [0] = acc_train [1] = acc_test
    avg_train_acc = []
    lasso_result = np.zeros((N, 2))
    dlda_result = np.zeros((N, 2))
    for i in range(0, N):
        X, y = generate_data() # write the database to file?
        X_train, y_train, X_test, y_test = split_data(X, y)
        lasso_result[i][0], lasso_result[i][1] = train_test_lasso(alpha, X_train, y_train, X_test, y_test)
        dlda_result[i][0], dlda_result[i][1] = train_test_dlda(delta, X_train, y_train, X_test, y_test)
    avg_train_acc.append(np.mean(lasso_result, axis = 0))
    avg_train_acc.append(np.mean(dlda_result, axis = 0))
    return avg_train_acc


# can further write it to config.json and loop for different configuration
# parameters
num_obs = 25; # number of observations
num_predictors = 100; # number of predictors
prob = 0.5 # probability that yi = 1
beta = np.random.uniform(0, 1, size=(1, num_predictors))# signal of each predictor
# multivariate normal distribution
sigma = np.eye(num_predictors) # covariance matrix
mu = np.zeros(num_predictors) # ith row for mu
# parameters for lasso model
a = 0.1 # for regularization
# instances of data
N = 100
# for dlda
d = 0.1