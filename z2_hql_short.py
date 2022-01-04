import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

input_layer_size  = 400
num_labels = 10

data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size

def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    n = theta.size
    J = 0
    grad = np.zeros(theta.shape)
    h = utils.sigmoid(np.dot(X, theta))
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))) + (lambda_/(2*m)) * (np.dot(theta[1:].T, theta[1:]))
    grad[0] = (1/m) * (np.dot(X[:, 0].T, h-y))
    for i in range(1, n):
        grad[i] = (1/m) * (np.dot(X[:, i].T, h-y)) + (lambda_/m) * (theta[i])
    return J, grad

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for c in range(num_labels):
        initial_theta = np.zeros(n+1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,
                               initial_theta,
                               (X, (y==c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)
        all_theta[c] = res.x
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros(m)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    Z = np.dot(X, all_theta.T)
    for i in range(m):
        p[i] = np.argmax(Z[i], axis = 0)
    return p

pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))