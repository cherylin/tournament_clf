#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To explore the effects of beta on the accuracy of models and also the coefficients of the models
In this case, we fixed beta's entries as between [-1, 1], but controling how many non-zero entries are there in the 
beta, namely the sparsity of beta.
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


def evaluation(all_density, all_X, all_y, N, p, a, d, c, n, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for density in all_density:
        num_train = math.ceil(n*0.7)
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[density][i]
            y = all_y[density][i]
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
        print('======density = {} ======'.format(density))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))
    
    f = plt.figure(figsize=(15,5))
    plt.plot(all_density, train_acc, 'bo-', label='training accuracy')
    plt.plot(all_density, test_acc, 'ro-', label='test accuracy')
    plt.xlabel('density of non-zero entries in beta')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('{}, n = 100, p = 200 vs sparsity of beta'.format(model))
    f.savefig('./outputs/beta_impact_sparsity/plot_{}.png'.format(model))

    with open('./outputs/beta_impac_sparsity/coef_{}'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('density={},'.format(all_density[i]))
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
    all_density = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]
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
    X_in_all_density = {}
    y_in_all_density = {}
    for d in all_density:    
        all_X = []
        all_y = []
        for j in range(0, N):
            # assume unfiromly distributied in [0, 1]
            beta = scipy.sparse.random(1, p, density=d, dtype=float, random_state=None, data_rvs=scipy.stats.uniform(loc=-1, scale=2).rvs)
            if (j+1) % 20 == 0:
                with open ('./outputs/beta_impact_sparsity/generated_beta_density={}_{}th-example.csv'.format(d, j+1),'w') as file:
                    file.write(','.join(map(str, beta.A)))
            X, y = generate_data(n, prob, mu, sigma, beta.A)
            all_X.append(X)
            all_y.append(y)
        X_in_all_density[d] = all_X
        y_in_all_density[d] = all_y
    models = ['lasso', 'dlda', 'svm', 'tc']
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(all_density, X_in_all_density, y_in_all_density, N, p, a, d, c, n, model=m)
        result[m] = (train_acc, test_acc)
    with open('./outputs/beta_impact_sparsity/accuracy_comparison.csv', 'w') as fout:
        fout.write(','.join(['density of non-zero entries in beta','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test', 'tc-train', 'tc-test']))
        fout.write('\n') 
        for i, d in enumerate(all_density):
            output_list = [
                d,
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