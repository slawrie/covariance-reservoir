''' This file contains all the functions to train mean/covariance based linear readouts.
Mean readouts are implemented with Ridge, scikit-learn. 
Covariance readouts are implemented following the gradient descent rule derived in 
https://dx.plos.org/10.1371/journal.pcbi.1008127, by Gilson et al.
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


# auxiliary functions
def my_covariance(x):
    N = x.shape[2]
    m1 = x - x.sum(2, keepdims=1) / N
    out = np.einsum('ijk,ilk->ijl', m1, m1) / (N - 1)
    return out

def seq2mean(y, num_classes):
    # map class label list to vectors
    target_means = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        target = np.zeros(num_classes)
        target[int(y[i])] = 1
        target_means[i, :] = target
    return target_means

def seq2cov(y, num_classes):
    # map class label list to covariances
    target_covariances = np.zeros((len(y), num_classes, num_classes))
    for i in range(len(y)):
        target_covariances[i, int(y[i]), int(y[i])] = 1
    return target_covariances


class MeanOptimizers:
    def __init__(self, params, batch_size=None):
        self.params = params
        self.batch_size = batch_size

class RidgeReg(MeanOptimizers):
    def __init__(self, params, batch_size=None, solver='auto', alpha=0.1, fit_intercept=True, num_classes=2):
        super().__init__(params, batch_size)
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.num_classes = num_classes

    def fit(self, X, y):
        means_X = np.mean(X, axis=2)
        target_means = seq2mean(y, self.num_classes)
        reg = Ridge(fit_intercept=self.fit_intercept,
                    alpha=self.alpha, solver=self.solver).fit(means_X, target_means)

        # update parameters
        self.params[:, 0] = reg.intercept_
        self.params[:, 1:] = reg.coef_


class CovarianceOptimizers:
    def __init__(self, params, batch_size=None, num_classes=2):
        self.params = params
        self.batch_size = batch_size
        self.num_classes = num_classes
        # Set basis of U matrices
        self.U = np.zeros((self.params.shape[0] * self.params.shape[1], self.params.shape[0], self.params.shape[1]))
        self.UT = np.zeros((self.params.shape[0] * self.params.shape[1], self.params.shape[1], self.params.shape[0]))
        m = 0
        for i in range(self.params.shape[0]):
            for j in range(self.params.shape[1]):
                self.U[m, i, j] = 1
                self.UT[m, j, i] = 1
                m += 1

    def get_loss(self, X, y):
        # X: examples, inputs, time
        # y: labels, start from 0
        # create target matrices
        self.batch_size = X.shape[0]
        target_covariances = seq2cov(y, self.num_classes)

        # add bias unit to reservoir
        ones = np.ones((X.shape[0], 1, X.shape[2]))
        u_ = np.concatenate((ones, X), axis=1)
        outputs = np.einsum('ij,kjt -> kit', self.params, u_)
        # calculate covariance
        out_covariances = my_covariance(outputs)
        # mask off diagonal elements
        identity = np.identity(self.num_classes)
        mask = np.where(identity == 1)
        Qerror = np.zeros((out_covariances.shape[0], out_covariances.shape[1], out_covariances.shape[2]))

        Qerror[:, mask[0], mask[1]] = np.subtract(
            target_covariances[:, mask[0], mask[1]],
            out_covariances[:, mask[0], mask[1]])  # numSamples, cov, cov
        return Qerror

    def get_gradients(self, X, Qerror):
        # Now get matrix of derivatives per sample and weight
        # add bias unit to reservoir
        ones = np.ones((X.shape[0], 1, X.shape[2]))
        u_ = np.concatenate((ones, X), axis=1)
        covariances_X = my_covariance(u_)
        # Covariance matrix needs to be expanded to include the bias covariance: update the resCovariance directly
        PB = np.einsum('lij, kj -> lik', covariances_X, self.params)  # (examples, res, out)
        BP = np.einsum('im, lmj -> lij', self.params, covariances_X)  # (examples, out, res)

        dQ = np.einsum('mij, ljk -> mlik', self.U, PB) \
        + np.einsum('ljk, mki -> mlji', BP, self.UT)  # weights, samples, cov, cov

        delta = np.einsum('mij, kmij -> k', Qerror, dQ)  # vector of gradients over batch for each weight
        delta = delta.reshape(self.params.shape)
        return delta

class GradientDescent(CovarianceOptimizers):
    # standard gradient descent update rule
    def __init__(self, params, batch_size=None, num_classes=2, lr=0.01):
        super().__init__(params, batch_size, num_classes)
        self.lr = lr

    def update_params(self, delta):
        self.params += self.lr * delta/self.batch_size

