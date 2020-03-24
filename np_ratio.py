#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To explore the effects of n vs p ratio to the accuracy of models.
We fix p = 200, n to be 9 different values. 
"""

from models import *
import numpy as np
import mlpy
from sklearn import linear_model, metrics
from sklearn.svm import LinearSVC
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from numpy import random
np.random.seed(1)
import math
import matplotlib.pyplot as plt

def generate_data(num_obs, prob, mu, sigma, beta):
    """Generate X, y based on the chosen generationg process, may play around different generation process."""
    # (1) generate Y dim = n*1
    y = np.random.choice(a=[-1, 1], size=(num_obs,1), replace=True, p=[1-prob, prob])
    # (2) generate Xi
    # generate error terms dim = n*p 
    U = np.random.multivariate_normal(mean=mu, cov=sigma, size=num_obs)
    # multiply yi's and beta and add with U to get X, dimX = n*p
    X = np.dot(y, beta) + U
    return X, y


def evaluation(all_n, all_X, all_y, N, p, a, d, c, model='lasso'):
    # sigma = np.diagflat(np.random.uniform(0, 2, size=(1, num_predictors)))
    train_acc = []
    test_acc = []
    coefficients = []
    for n in all_n:
        num_train = math.ceil(n*0.7)
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[n][i]
            y = all_y[n][i]
            X_train, X_test = X[0:num_train], X[num_train:n]
            y_train, y_test = y[0:num_train], y[num_train:n]
            if model == 'lasso':
                accuracy[i][0], accuracy[i][1], coefs[i] = lasso(a, X_train, y_train, X_test, y_test)
            elif model == 'dlda':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = dlda(d, X_train, y_train, X_test, y_test)
            elif model == 'svm':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = linearsvm(c, X_train, y_train, X_test, y_test)
            elif model == 'tc':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = tournament_classifier(X_train, y_train, X_test, y_test)
        accuracy = np.mean(accuracy, axis = 0)
        train_acc.append(accuracy[0])
        test_acc.append(accuracy[1])
        coefficients.append(np.mean(coefs, axis = 0))
        print('======n = {} p = 200 ======'.format(n))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))
    
    f = plt.figure(figsize=(15,5))
    plt.plot(all_n, train_acc, 'bo-', label='training accuracy')
    plt.plot(all_n, test_acc, 'ro-', label='test accuracy')
    plt.xlabel('number of observatoins')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('{}, p = 200'.format(model))
    f.savefig('./outputs/np_ratio/plot_{}.png'.format(model))

    with open('./outputs/coef_np_ratio_{}.csv'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('n={},'.format(all_n[i]))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc


def main():
    all_n= [20, 40, 50, 80, 100, 200, 300, 400, 500, 600]
    p = 200
    prob = 0.5
    beta = np.random.uniform(-1, 1, size=(1, p))
    mu = np.zeros(p)
    sigma = np.eye(p)
    N = 100
    # for regularization, alpha
    a = 0.1 
    # for dlda, delta
    d = 0.1
    # for SVM
    c = 0.5
    '''generate dataset'''
    X_in_all_n = {}
    y_in_all_n = {}
    for n in all_n:    
        all_X = []
        all_y = []
        for j in range(0, N):
            X, y = generate_data(n, prob, mu, sigma, beta)
            all_X.append(X)
            all_y.append(y)
        X_in_all_n[n] = all_X
        y_in_all_n[n] = all_y
    models = ['lasso', 'dlda', 'svm', 'tc']
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(all_n, X_in_all_n, y_in_all_n, N, p, a, d, c, model=m)
        result[m] = (train_acc, test_acc)
    with open('./outputs/accuracy_np_ratio.csv', 'w') as fout:
        fout.write(','.join(['n','p','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test','tc-train','tc-test']))
        fout.write('\n') 
        for i, n in enumerate(all_n):
            output_list = [
                n,
                200,
                result['lasso'][0][i], 
                result['lasso'][1][i], 
                result['dlda'][0][i], 
                result['dlda'][1][i], 
                result['svm'][0][i], 
                result['svm'][1][i],
                result['tc'][0][i],
                result['tc'][1][i]
            ]
            fout.write(','.join(map(str, output_list)))
            fout.write('\n')
        
if __name__ == "__main__":
    main()