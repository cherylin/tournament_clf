#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting 8: Imbalanced class
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
import argparse
import sys
from generate_data import generate_data

# Parameter setting
# this is the part we FIX
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
# The parameters that we change
probs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
OUTPUT_DIR = './outputs/setting_8/'

def evaluation(probs, all_X, all_y, N, p, model='lasso'):
    train_acc = []
    test_acc = []
    coefficients = []
    for prob in probs:
        num_train = math.ceil(PARAMS['n']*PARAMS['train_percent'])
        accuracy = np.zeros((N, 2))
        coefs = np.zeros((N, p))
        for i in range(0, N):
            X = all_X[prob][i]
            y = all_y[prob][i]
            X_train, X_test = X[0:num_train], X[num_train:PARAMS['n']]
            y_train, y_test = y[0:num_train], y[num_train:PARAMS['n']]
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
        print('======prob = {} ======'.format(prob))
        print("{}: train accuracy= {} and test accuracy= {}".format(model, accuracy[0], accuracy[1]))
    
    # plot the accuracy data with respect to n for each model
    f = plt.figure(figsize=(15,5))
    plt.plot(probs, train_acc, 'bo-', label='training accuracy')
    plt.plot(probs, test_acc, 'ro-', label='test accuracy')
    plt.xlabel('probability of Y=1')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('probability of Y=1 vs accuracy for {}'.format(model))
    f.savefig(OUTPUT_DIR+'plot_{}.png'.format(model))

    with open(OUTPUT_DIR+'coef_np_ratio_{}.csv'.format(model), 'w') as file:
        for i, coef in enumerate(coefficients):
            file.write('prob={},'.format(probs[i]))
            file.write(','.join(map(str, coef)))
            file.write('\n')
    return train_acc, test_acc

def main():
    '''generate dataset'''
    X_in_all_probs = {}
    y_in_all_probs = {}
    for prob in probs:
        all_X = []
        all_y = []
        for j in range(0, N):
            X, y = generate_data(PARAMS['n'], prob, mu, sigma, beta)
            all_X.append(X)
            all_y.append(y)
        X_in_all_probs[prob] = all_X
        y_in_all_probs[prob] = all_y

    result = {}
    for m in models:
        train_acc, test_acc = evaluation(probs, X_in_all_probs, y_in_all_probs, N, p, model=m)
        result[m] = (train_acc, test_acc)
    with open(OUTPUT_DIR+'accuracy_np_ratio.csv', 'w') as fout:
        fout.write(','.join(['prob','lasso-train','lasso-test','dlda-train','dlda-test','svm-train','svm-test','tc-train','tc-test']))
        fout.write('\n') 
        for i, prob in enumerate(probs):
            output_list = [
                prob,
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