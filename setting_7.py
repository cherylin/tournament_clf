#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting 7: Combine setting 5 and setting 6

We try exhaustive combinations between alpha and (smalles_coef, largest_coef) tuple. We try alpha in [0.1, 0.9] and (smalles_coef, largest_coef) in [(-1, 1), (-10, 10), (-1, 20), (-5, 1)]
"""

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
all_alpha = [0.1, 0.9]
all_ranges = [(-1, 1), (-10, 10), (-1, 20), (-5, 1)]
combinations = []
for d in all_alpha:
    for r in all_ranges:
        comb = {'alpha': d, 'a': r[0],'b':r[1]}
        combinations.append(comb)
OUTPUT_DIR = './outputs/setting_7/'

def evaluation(combinations, all_X, all_y, N, p, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for k, comb in enumerate(combinations):
        num_train = math.ceil(PARAMS['n']*PARAMS['train_percent'])
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[k][i]
            y = all_y[k][i]
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
        print('======combination: alpha = {}, a = {}, b = {} ======'.format(comb['alpha'], comb['a'], comb['b']))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))

    with open(OUTPUT_DIR+'coef_{}.txt'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('alpha = {}, min value={}, max value={},'.format(combinations[i]['alpha'], combinations[i]['a'], combinations[i]['b']))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc


def main():
    '''generate dataset'''
    X_in_all_combs = {}
    y_in_all_combs = {}
    for i, comb in enumerate(combinations):    
        all_X = []
        all_y = []
        for j in range(0, N):
            sigma = make_sparse_spd_matrix(p, alpha=comb['alpha'], smallest_coef=comb['a'], largest_coef=comb['b'], norm_diag=False)
            if (j+1) % 20 == 0:
                with open (OUTPUT_DIR+'generated_sigma_combination={}_{}th-example.csv'.format(i+1, j+1),'w') as file:
                    file.write(','.join(map(str, beta)))
            X, y = generate_data(n, PARAMS['prob'], mu, sigma, beta)
            all_X.append(X)
            all_y.append(y)
        X_in_all_combs[i] = all_X
        y_in_all_combs[i] = all_y
    result = {}
    for m in models:
        train_acc, test_acc = evaluation(combinations, X_in_all_combs, y_in_all_combs, N, p, model=m)
        result[m] = (train_acc, test_acc)
    with open(OUTPUT_DIR+'accuracy_comparison.csv', 'w') as fout:
        fout.write(','.join(['alpha, min val of entries', 'max val of entries','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test', 'tc-train', 'tc-test']))
        fout.write('\n') 
        for i, comb in enumerate(combinations):
            output_list = [
                comb['alpha'],
                comb['a'],
                comb['b'],
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