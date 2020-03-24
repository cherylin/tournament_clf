#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To explore the effects of beta on the accuracy of models and also the coefficients of the models
In this case, we fixed the sparsity of beta, but controling the entries of beta.
We inspect how everything changes, when the entries of beta is getting larger.
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
import scipy

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


def evaluation(all_ranges, all_X, all_y, N, p, a, d, c, n, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for r in all_ranges:
        num_train = math.ceil(n*0.7)
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[r[1]][i]
            y = all_y[r[1]][i]
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
        print('======density=0.2, max_val={} ======'.format(r[1]))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))
    
    f = plt.figure(figsize=(15,5))
    max_vals = [r[1] for r in all_ranges]
    plt.plot(max_vals, train_acc, 'bo-', label='training accuracy')
    plt.plot(max_vals, test_acc, 'ro-', label='test accuracy')
    plt.xlabel('maximum value of weights on beta')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('{}, n = 100, p = 200 vs weights on beta'.format(model))
    f.savefig('./outputs/beta_impact_weight/plot_{}.png'.format(model))

    with open('./outputs/beta_impact_weight/coef_{}.txt'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('min value={}, max value={},'.format(all_ranges[i][0], all_ranges[i][1]))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc


def main():
    n = 100
    p = 200
    prob = 0.5
    '''
    Change beta !!!
    beta = np.random.uniform(-1, 1, size=(1, p))
    '''
    all_ranges = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]
    sigma = np.eye(p)
    mu = np.zeros(p)
    N = 100
    # for regularization, alpha
    a = 0.1 
    # for dlda, delta
    d = 0.1
    # for SVM
    c = 0.5
    '''generate dataset'''
    X_in_all_ranges = {}
    y_in_all_ranges = {}
    for r in all_ranges:    
        all_X = []
        all_y = []
        for j in range(0, N):
            beta = scipy.sparse.random(1, p, density=0.2, dtype=float, random_state=None, data_rvs=scipy.stats.uniform(loc=r[0], scale=r[1]).rvs)
            if (j+1) % 20 == 0:
                with open ('./outputs/beta_impact_weight/generated_beta_range={}_{}th-example.csv'.format(r, j+1),'w') as file:
                    file.write(','.join(map(str, beta.A)))
            X, y = generate_data(n, prob, mu, sigma, beta.A)
            all_X.append(X)
            all_y.append(y)
        X_in_all_ranges[r[1]] = all_X
        y_in_all_ranges[r[1]] = all_y
    models = ['lasso', 'dlda', 'svm', 'tc']
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(all_ranges, X_in_all_ranges, y_in_all_ranges, N, p, a, d, c, n, model=m)
        result[m] = (train_acc, test_acc)
    with open('./outputs/beta_impact_weight/accuracy_comparison.csv', 'w') as fout:
        fout.write(','.join(['min val of entries', 'max val of entries','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test', 'tc-train', 'tc-test']))
        fout.write('\n') 
        for i, r in enumerate(all_ranges):
            output_list = [
                r[0],
                r[1],
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