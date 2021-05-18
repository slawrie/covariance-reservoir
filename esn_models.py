'''
This file contains all the elements to define a reservoir, run it and collect states
For training, you should create a reservoir, run it with the data, collect states and then use
a linear readout to create the mapping.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

# auxiliary function to compute covariance for tensors
def my_covariance(x):
    N = x.shape[2]
    m1 = x - x.sum(2, keepdims=1) / N
    out = np.einsum('ijk,ilk->ijl', m1, m1) / (N - 1)
    return out

class SequentialReservoir:
    def __init__(self, inSize, resSize, outSize, style='random', leak=1.0, in_density=1.0, density=1.0, radius=0.9,
                 random_state=42):
        self.random_state = random_state 
        np.random.seed(self.random_state) # fix seed to fix parameters
        self.inSize = inSize  # number of inputs 
        self.resSize = resSize # neurons in reservoir
        self.outSize = outSize # number of outputs, must match number of classes
        self.density = density # connection density within reservoir
        self.radius = radius # spectral radius
        self.in_density = in_density # connection density from inputs to reservoir
        self.Win = (np.random.rand(self.resSize, self.inSize + 1)) - 0.5
        self.Win[np.random.rand(self.resSize, self.inSize + 1) > self.in_density] = 0
        self.style = style

        # now, get the adjacency matrices
        # feedforward
        if self.density == 0 or self.radius == 0:
            self.W = np.zeros((self.resSize, self.resSize))
            self.rhoW = 0
        else:
            if style == 'random':
                self.W = np.random.rand(self.resSize, self.resSize) - 0.5  # non sparse
                self.W[np.random.rand(resSize, resSize) > self.density] = 0
            elif style == 'sym':
                self.W = np.zeros([self.resSize, self.resSize])
                for i in range(self.resSize):
                    for j in range(i):
                        if np.random.rand() < self.density:
                            self.W[i, j] = np.random.rand() - 0.5
                            self.W[j, i] = self.W[i, j]
                            
            elif style == 'skewsym':
                self.W = np.zeros([self.resSize, self.resSize])  
                for i in range(self.resSize):
                    for j in range(i):
                        if np.random.rand() < self.density:
                            self.W[i, j] = np.random.rand() - 0.5
                            self.W[j, i] = -self.W[i, j]
            
            elif style == 'self_rec': #only self connections, self recurrent
                self.W = np.zeros([self.resSize, self.resSize])  # weights between 0 and 1
                for i in range(self.resSize):
                        self.W[i, i] = np.random.rand() - 0.5

            self.rhoW = np.max(abs(scipy.linalg.eig(self.W)[0]))
            self.W *= self.radius / self.rhoW # scale with spectral radius

        self.Wout = np.random.rand(self.outSize, self.resSize + 1) - 0.5
        self.leak = leak
        self.resStates = None
        self.resCovariance = None
        self.outStates = None
        self.outCovariance = None
        self.resMean = None
        self.outMean = None
        return


    def run(self, data, initLen, trainLen, covariance=False, mean=False):
        '''Data is an array. Dimension is (numExamples, numInputs, timeLen)'''
        self.resStates = np.zeros((data.shape[0], self.resSize, trainLen))  # collected states
        self.outStates = np.zeros((data.shape[0], self.outSize, trainLen))  # output units states

        # run the reservoir with the data and collect X
        x = np.zeros((data.shape[0], self.resSize))  # current state of reservoir
        y = np.zeros((data.shape[0], self.outSize))  # current state of outputs

        # add bias unit to input data
        ones = np.ones((data.shape[0], 1, data.shape[2]))
        inputs = np.concatenate((ones, data), axis=1)
        for t in range(trainLen + initLen):
            u = inputs[:, :, t]  # this has shape batch, inputs
            x = (1 - self.leak) * x + self.leak * np.tanh(np.einsum('ij, kj ->ik', u, self.Win) \
                                                          + np.einsum('kj, ij -> ik', self.W, x))  # batch, res

            # add bias to reservoir
            ones = np.ones((data.shape[0], 1))
            u_ = np.concatenate((ones, x), axis=1)
            y = np.einsum('ij,kj -> ki', self.Wout, u_)
            if t >= initLen:
                self.resStates[:, :, t - initLen] = x
                self.outStates[:, :, t - initLen] = y

        if covariance:  # update covariances
            self.resCovariance = np.zeros((data.shape[0], self.resSize + 1, self.resSize + 1))
            self.outCovariance = np.zeros((data.shape[0], self.outSize, self.outSize))
            ones = np.ones((self.resStates.shape[0], 1, self.resStates.shape[2]))
            states = np.concatenate((ones, self.resStates), axis=1)
            self.resCovariance = my_covariance(states)
            self.outCovariance = my_covariance(self.outStates)

        # update mean states
        if mean:  # update mean states
            self.resMean = np.mean(self.resStates, axis=2)
            self.outMean = np.mean(self.outStates, axis=2)
        return

    def update_outputs(self, trainLen, initLen, mean=False, covariance=False):
        '''Use this function to only update output states and covariances during training'''
        # run the reservoir with the data and collect X
        y = np.zeros((self.resStates.shape[0], self.outSize))  # current state of outputs

        # add bias unit to input data
        # add bias to reservoir
        ones = np.ones((self.resStates.shape[0], 1, initLen + trainLen))
        u_ = np.concatenate((ones, self.resStates), axis=1)  # examples, units, time

        self.outStates = np.einsum('ij,kjt -> kit', self.Wout, u_)
        if covariance:
            self.outCovariance = my_covariance(self.outStates)
        if mean:
            self.outMean = np.mean(self.outStates, axis=2)
        return


    def predict(self, mode='mean'):
        #Run data through reservoir, get covariances in output units. If var0/mean0 > var 1/mean1, class is 0.
        Y = []
        if mode == 'mean':
            for ex in range(self.resStates.shape[0]):
                max_out = np.max(self.outMean[ex, :])
                pred = np.where(self.outMean[ex, :] == max_out)[0][0]
                Y.append(pred)

        if mode == 'covariance':
            for ex in range(self.resStates.shape[0]):
                diagonals = np.diag(self.outCovariance[ex, :, :])
                max_out = np.max(diagonals)
                pred = np.where(diagonals == max_out)[0][0]
                Y.append(pred)
        return Y


    def score(self, Y_true, Y_pred):
        return accuracy_score(Y_true, Y_pred)



class SegregatedReservoir:
    # create a reservoir with segregated inputs and outputs
    def __init__(self, inSize, resSize, outSize, style='random', leak=1.0, in_density=1.0, density=1.0, radius=0.9,
                 random_state=42, Nin = 50, Nout = 50):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.density = density
        self.radius = radius
        self.in_density = in_density
        self.Nin = Nin
        self.Nout = Nout
        self.style = style

        # segregate input nodes
        self.Win = np.zeros([self.resSize, self.inSize + 1])
        for i in range(int(Nin/2)):
            self.Win[i, :] = np.random.rand(self.inSize + 1) - 0.5
            self.Win[self.resSize-1-i, :] = np.random.rand(self.inSize + 1) - 0.5

        # now, get the connectivity matrices
        if style == 'random':
            self.W = np.random.rand(self.resSize, self.resSize)  # non sparse
            self.W = self.W - 0.5  # weights between -0.5 and  0.5
            self.W[np.random.rand(resSize, resSize) > self.density] = 0

        elif style == 'sym':
            self.W = np.zeros([self.resSize, self.resSize])
            for i in range(self.resSize):
                for j in range(i):
                    if np.random.rand() < self.density:
                        self.W[i, j] = np.random.rand() - 0.5
                        self.W[j, i] = self.W[i, j]

        elif style == 'skewsym':
            self.W = np.zeros([self.resSize, self.resSize]) 
            for i in range(self.resSize):
                for j in range(i):
                    if np.random.rand() < self.density:
                        self.W[i, j] = np.random.rand() - 0.5
                        self.W[j, i] = -self.W[i, j]


        self.rhoW = np.max(abs(scipy.linalg.eig(self.W)[0]))
        self.W *= self.radius / self.rhoW

        # segregate outputs when running reservoir, resStates will now only contain information about Nout nodes
        self.Wout = np.random.rand(self.outSize, self.Nout + 1) - 0.5
        self.leak = leak
        self.resStates = None
        self.resCovariance = None
        self.outStates = None
        self.outCovariance = None
        self.resMean = None
        self.outMean = None
        return

    def run(self, data, initLen, trainLen, covariance=False, mean=False):
        '''Data is an array. Dimension is (numExamples, numInputs, timeLen)'''
        self.resStates = np.zeros((data.shape[0], self.Nout, trainLen))  # collected states
        self.outStates = np.zeros((data.shape[0], self.outSize, trainLen))  # output units states

        # run the reservoir with the data and collect X
        x = np.zeros((data.shape[0], self.resSize))  # current state of reservoir
        y = np.zeros((data.shape[0], self.outSize))  # current state of outputs

        # add bias unit to input data
        ones = np.ones((data.shape[0], 1, data.shape[2]))
        inputs = np.concatenate((ones, data), axis=1)
        for t in range(trainLen + initLen):
            u = inputs[:, :, t]  # this has shape batch, inputs
            x = (1 - self.leak) * x + self.leak * np.tanh(np.einsum('ij, kj ->ik', u, self.Win) \
                                                          + np.einsum('kj, ij -> ik', self.W, x))  # batch, res

            # to update outputs only use Nout nodes
            ones = np.ones((data.shape[0], 1))
            u_ = np.concatenate((ones, x[:, int(self.resSize/2 - self.Nout/2):int(self.resSize/2 + self.Nout/2)]), axis=1)
            y = np.einsum('ij,kj -> ki', self.Wout, u_)
            if t >= initLen:
                self.resStates[:, :, t - initLen] = x[:, int(self.resSize/2 - self.Nout/2):int(self.resSize/2 + self.Nout/2)]
                self.outStates[:, :, t - initLen] = y

        if covariance:  # update covariances
            self.resCovariance = np.zeros((data.shape[0], self.Nout + 1, self.Nout + 1))
            self.outCovariance = np.zeros((data.shape[0], self.outSize, self.outSize))
            ones = np.ones((self.resStates.shape[0], 1, self.resStates.shape[2]))
            states = np.concatenate((ones, self.resStates), axis=1)
            self.resCovariance = my_covariance(states)
            self.outCovariance = my_covariance(self.outStates)

        # update mean states
        if mean:  # update mean states
            self.resMean = np.mean(self.resStates, axis=2)
            self.outMean = np.mean(self.outStates, axis=2)
        return

    def update_outputs(self, trainLen, initLen, mean=False, covariance=False):
        '''Use this function to only update output states and covariances during training'''
        # run the reservoir with the data and collect X
        y = np.zeros((self.resStates.shape[0], self.outSize))  # current state of outputs

        # add bias unit to input data
        # add bias to reservoir
        ones = np.ones((self.resStates.shape[0], 1, initLen + trainLen))
        u_ = np.concatenate((ones, self.resStates), axis=1)  # examples, units, time

        self.outStates = np.einsum('ij,kjt -> kit', self.Wout, u_)
        if covariance:
            self.outCovariance = my_covariance(self.outStates)
        if mean:
            self.outMean = np.mean(self.outStates, axis=2)
        return

    def predict(self, mode='mean'):
        Y = []
        if mode == 'mean':
            for ex in range(self.resStates.shape[0]):
                max_out = np.max(self.outMean[ex, :])
                pred = np.where(self.outMean[ex, :] == max_out)[0][0]
                Y.append(pred)

        if mode == 'covariance':
            for ex in range(self.resStates.shape[0]):
                diagonals = np.diag(self.outCovariance[ex, :, :])
                max_out = np.max(diagonals)
                pred = np.where(diagonals == max_out)[0][0]
                Y.append(pred)
        return Y

    def score(self, Y_true, Y_pred):
        return accuracy_score(Y_true, Y_pred)

