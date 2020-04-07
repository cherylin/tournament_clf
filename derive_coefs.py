from models import *
import numpy as np
from sklearn import linear_model, metrics
from numpy import random
np.random.seed(1)
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import lsqr

def read_matrix(filename):
    matrix = []
    with open(filename, 'r') as fin:
        rows = fin.read().strip().split('\n')
        for row in rows:
            entries = row.split(',')
            matrix.append(list(map(float, entries)))
    return np.array(matrix)

def prob_y_given_x(X, y, sigma, beta):
    # Assume p(y=1) and p(y=-1) are 1/2
    prob_1_given_x = []
    prob_2_given_x = []
    for x in X:
        res1 = multivariate_normal.pdf(x, mean=beta[0], cov=sigma)
        res2 = multivariate_normal.pdf(x, mean=-beta[0], cov=sigma)
        prob_1_given_x.append(res1/(res1+res2))
        prob_2_given_x.append(res2/(res1+res2))
    return np.array(prob_1_given_x), np.array(prob_2_given_x)

def main():
    # all data are 2-dimensional matrix
    X = read_matrix('./var/X_data.csv')
    y = read_matrix('./var/y_label.csv')
    sigma = read_matrix('./var/sigma_data.csv')
    beta = read_matrix('./var/beta_data.csv')
    n, p = X.shape
    # for each x, compute its probability of P(y|x)
    # y = 1, X ~ MVN(beta, sigma)
    # y = -1, X ~ MVN(-beta, sigma)
    prob_pos, prob_neg = prob_y_given_x(X, y, sigma, beta)
    pred = np.where(prob_pos > 0.5, 1, -1)
    pred = pred[:, np.newaxis]
    prob_pos = prob_pos[:, np.newaxis]
    # approximate the optimal coefficients
    # prob_pos should be the score return by the classifier
    # extra_col = np.ones(shape=(n, 1))
    # X = np.append(X, extra_col, axis=1)
    # solve least square solution for Xw = prob_pos 
    w, *_ = lsqr(X, prob_pos)
    print(w)
    np.savetxt('./var/optimal_w.csv', w, delimiter=',')

if __name__ == "__main__":
    main()