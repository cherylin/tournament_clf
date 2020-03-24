"""
All the training and testing on different models
"""
import numpy as np
import mlpy
from sklearn import linear_model, metrics
from sklearn.svm import LinearSVC
from tc import tc, tc_pred

def lasso(alpha, X_train, y_train, X_test, y_test):
    """
    Lasso model: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    # calculate training accuracy and test accuracy
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    y_pred_test[y_pred_test > 0] = 1
    y_pred_test[y_pred_test <= 0] = -1
    y_pred_train[y_pred_train > 0] = 1
    y_pred_train[y_pred_train <= 0] = -1
    training_acc = metrics.accuracy_score(y_train, y_pred_train)
    testing_acc = metrics.accuracy_score(y_test, y_pred_test)
    return training_acc, testing_acc, np.array(clf.coef_)

def dlda(delta, X_train, y_train, X_test, y_test):
    clf = mlpy.DLDA(delta=delta)
    y_train = np.transpose(y_train)[0]
    clf.learn(X_train, y_train)
    y_pred_train = clf.pred(X_train)
    y_pred_test = clf.pred(X_test)
    coef = np.mean(clf._dprime, axis=0)
    training_acc = metrics.accuracy_score(y_train, y_pred_train)
    testing_acc = metrics.accuracy_score(y_test, y_pred_test)
    return training_acc, testing_acc, np.array(coef)

def linearsvm(c, X_train, y_train, X_test, y_test):
    clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=c, class_weight='balanced')
    y_train = np.transpose(y_train)[0]
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    training_acc = metrics.accuracy_score(y_train, y_pred_train)
    testing_acc = metrics.accuracy_score(y_test, y_pred_test)
    return training_acc, testing_acc, np.array(clf.coef_)

def tournament_classifier(X_train, y_train, X_test, y_test):
    # iter = 20
    coefs, train_preds, train_acc = tc(X_train, np.squeeze(y_train), 20)
    test_preds, test_acc = tc_pred(X_test, np.squeeze(y_test), coefs)
    assert(metrics.accuracy_score(y_train, train_preds) == train_acc)
    assert(metrics.accuracy_score(y_test, test_preds) == test_acc)
    return train_acc, test_acc, np.array(coefs)