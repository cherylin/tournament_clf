#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To explore the effects of covariance matrix on the ratio to the accuracy of models.
Here, we control every other parameters, but sigma allows different predictors to have correlations
with each other. We change the number of dependency there are among predictors. Using the sparsity of sigma, 
we control how many predictros having correlations with each others. 
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


def evaluation(all_alpha, all_X, all_y, N, p, a, d, c, n, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for a in all_alpha:
        num_train = math.ceil(n*0.7)
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[a][i]
            y = all_y[a][i]
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
        print('======alpha = {} ======'.format(a))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))
    
    f = plt.figure(figsize=(15,5))
    plt.plot(all_alpha, train_acc, 'bo-', label='training accuracy')
    plt.plot(all_alpha, test_acc, 'ro-', label='test accuracy')
    plt.xlabel('alpha, probability that an entry is 0')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('{}, n = 100, p = 200 vs sparsity of covariance'.format(model))
    f.savefig('./outputs/cov_impact_correlation/plot_{}.png'.format(model))

    with open('./outputs/cov_impact_correlation/coef_{}.csv'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('alpha={},'.format(all_alpha[i]))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc


def main():
    n = 100
    p = 200
    prob = 0.5
    beta = np.random.uniform(-1, 1, size=(1, p))
    with open ('./outputs/cov_impact_correlation/coef_original_beta.csv','w') as file:
        file.write(','.join(map(str, beta[0])))
    mu = np.zeros(p)
    '''
    Change sigma !!!
    sigma = np.eye(p)
    sigma = np.diagflat(np.random.uniform(0, 2, size=(1, num_predictors)))
    '''
    # alpha = prob that the entry is 0
    all_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N = 100
    # for regularization, alpha
    a = 0.1
    # for dlda, delta
    d = 0.1
    # for SVM
    c = 0.5
    '''generate dataset'''
    X_in_all_alpha = {}
    y_in_all_alpha = {}
    for a in all_alpha:    
        all_X = []
        all_y = []
        for j in range(0, N):
            # Generate a sparse symmetric definite positive matrix.
            sigma = make_sparse_spd_matrix(p, alpha=a, smallest_coef=0, largest_coef=1, norm_diag=True)
            if (j+1) % 20 == 0:
                sigma.tofile('./outputs/cov_impact_correlation/sigma_alpha={}_{}th-example.txt'.format(a, j+1), sep=",", format="%s")
            X, y = generate_data(n, prob, mu, sigma, beta)
            all_X.append(X)
            all_y.append(y)
        X_in_all_alpha[a] = all_X
        y_in_all_alpha[a] = all_y
    models = ['lasso', 'dlda', 'svm', 'tc']
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(all_alpha, X_in_all_alpha, y_in_all_alpha, N, p, a, d, c, n, model=m)
        result[m] = (train_acc, test_acc)
    with open('./outputs/cov_impact_correlation/accuracy_comparison.csv', 'w') as fout:
        fout.write(','.join(['alpha','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test','tc-train','tc-test']))
        fout.write('\n') 
        for i, a in enumerate(all_alpha):
            output_list = [
                a,
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