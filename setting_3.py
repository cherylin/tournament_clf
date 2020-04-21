#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting 3
"""

from models import lasso, dlda, linearsvm, tournament_classifier
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
from generate_data import generate_data

models = ['lasso', 'dlda', 'svm', 'tc']
PARAMS = {
    'n': 50,
    'p': 100,
    'prob': 0.5,
    'N': 100,
    'train_percent': 0.8
}
p = PARAMS['p']
mu = np.zeros(p)
sigma = np.eye(p)
beta = np.random.uniform(-1, 1, size=(1, p))
N = PARAMS['N']
n = PARAMS['n']
# controls the smallest coefficient and largest coefficients of beta 
all_ranges = [(-1, 1), (-3, 3), (-10, 10), (-0.1, 0.1), (0, 0.01), (-5, 0.001)]
OUTPUT_DIR = './outputs/setting_3/'

def evaluation(all_ranges, all_X, all_y, N, p, n, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for r in all_ranges:
        num_train = math.ceil(n*PARAMS['train_percent'])
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[r[1]][i]
            y = all_y[r[1]][i]
            X_train, X_test = X[0:num_train], X[num_train:n]
            y_train, y_test = y[0:num_train], y[num_train:n]
            if model == 'lasso':
                accuracy[i][0], accuracy[i][1], coefs[i] = lasso(X_train, y_train, X_test, y_test)
            elif model == 'dlda':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = dlda(X_train, y_train, X_test, y_test)
            elif model == 'svm':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = linearsvm(X_train, y_train, X_test, y_test)
            elif model == 'tc':
                accuracy[i][0], accuracy[i][1], coefs[i,:] = tournament_classifier(X_train, y_train, X_test, y_test)
        accuracy = np.mean(accuracy, axis = 0)
        train_acc.append(accuracy[0])
        test_acc.append(accuracy[1])
        coefficients.append(np.mean(coefs, axis = 0))
        print('======range: [{}, {}] ======'.format(r[0], r[1]))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))

    with open(OUTPUT_DIR+'coef_{}.txt'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('min value={}, max value={},'.format(all_ranges[i][0], all_ranges[i][1]))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc


def main():
    '''generate dataset'''
    X_in_all_ranges = {}
    y_in_all_ranges = {}
    for r in all_ranges:    
        all_X = []
        all_y = []
        for j in range(0, N):
            a = r[0]
            b = r[1]
            beta = np.random.uniform(a, b, size=(1, p))
            if (j+1) % 20 == 0:
                with open (OUTPUT_DIR+'generated_beta_range={}_{}th-example.csv'.format(r, j+1),'w') as file:
                    file.write(','.join(map(str, beta)))
            X, y = generate_data(n, PARAMS['prob'], mu, sigma, beta)
            all_X.append(X)
            all_y.append(y)
        X_in_all_ranges[r[1]] = all_X
        y_in_all_ranges[r[1]] = all_y
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(all_ranges, X_in_all_ranges, y_in_all_ranges, N, p, n, model=m)
        result[m] = (train_acc, test_acc)
    with open(OUTPUT_DIR+'accuracy_comparison.csv', 'w') as fout:
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