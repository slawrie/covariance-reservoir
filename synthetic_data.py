'''
This script contains all the code to generate the time series used in the classification tasks
We can save the class instantiation to keep all parameters, or only save defining patterns and data
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random
from scipy.linalg import expm


class MeanStruct():
    '''
    This class generates input time series where the patterns differ in mean
    '''

    def __init__(self, numClasses, numPatterns, numExamples, numInputs, trainLen, initLen, sigma=1, density=0.1,
                 noise=1):
        self.numClasses = numClasses
        self.numPatterns = numPatterns
        self.numExamples = numExamples
        self.numInputs = numInputs
        self.trainLen = trainLen
        self.initLen = initLen
        self.density = density # of patterns
        self.sigma = sigma # std of sampling for patterns
        self.noise = noise # std of the sampling noise
        self.p = np.zeros((numClasses, numPatterns, numInputs, trainLen + initLen)) # mean patterns

        for j in range(numClasses):
            for i in range(numPatterns):
                M = self.sigma * np.random.randn(numInputs) # dense
                M[np.random.rand(M.shape[0]) > self.density] = 0 # sparse
                for t in range(trainLen+initLen):
                    self.p[j, i, :, t] = M

        # for each pattern, create numExamples instances of the noise variable
        z_noise = self.noise*np.random.randn(numClasses, numPatterns, numExamples, numInputs, trainLen + initLen)
        # create matrix of input vectors
        data = np.zeros((numClasses, numPatterns, numExamples, numInputs, trainLen + initLen))
        self.classes = []
        for j in range(numClasses):
            for i in range(numPatterns):
                for k in range(numExamples):
                    data[j, i, k, :, :] = z_noise[j, i, k, :, :] + self.p[j, i, :, :]
                    self.classes.append(j)

        # flatten the data
        self.data = data.reshape(numPatterns * numClasses * numExamples, numInputs, trainLen + initLen)
        return
    

class SpatialCovStruct():
    '''
    This class generates input time series characterized by zero-lagged covariance patterns
    '''

    def __init__(self, numClasses=2, numPatterns=30, numExamples=500, numInputs=10, trainLen=20, initLen=0, density=0.1, sigma=1, random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.numClasses = numClasses
        self.numPatterns = numPatterns
        self.numExamples = numExamples
        self.numInputs = numInputs
        self.trainLen = trainLen
        self.initLen = initLen
        self.density = density # of W matrix
        self.sigma = sigma # std of sampling for W elements
        self.classes = []
        # get covariance matrix W to define each pattern in each class
        self.W = np.zeros((numClasses, numPatterns, numInputs, numInputs))
        for i in range(numPatterns):
            for j in range(numClasses):
                W_ = random(numInputs, numInputs, density=self.density, random_state = self.random_state)
                self.W[j, i, :, :] = self.sigma*W_.A

        # create instances of the noise variable for each example
        z_noise = np.random.randn(numClasses, numPatterns, numExamples, numInputs, trainLen + initLen)
        # create matrix of input vectors
        data = np.zeros((numClasses, numPatterns, numExamples, numInputs, trainLen + initLen))
        
        for j in range(numClasses):
            for i in range(numPatterns):
                for k in range(numExamples):
                    data[j, i, k, :, :] = self.W[j, i, :, :] @ z_noise[j, i, k, :, :]
                    self.classes.append(j)
        # flatten the data
        self.data = data.reshape(numPatterns * numClasses * numExamples, numInputs, trainLen + initLen)
        return

    
class MixedStruct():
    '''
    This class generates input time series where the patterns differ in zero-lagged covariance and mean.
    '''

    def __init__(self, numClasses, numPatterns, numExamples, numInputs, trainLen, initLen, sigma=1.0, density=0.1):
        self.numClasses = numClasses
        self.numPatterns = numPatterns
        self.numExamples = numExamples
        self.numInputs = numInputs
        self.trainLen = trainLen
        self.initLen = initLen
        self.density = density # of W patterns
        self.sigma = sigma # std of sampling for p patterns
        self.W = np.zeros((numClasses, numPatterns, numInputs, numInputs))
        self.p = np.zeros((numClasses, numPatterns, numInputs, trainLen + initLen))
        for i in range(numPatterns):
            for j in range(numClasses):
                W_ = random(numInputs, numInputs, density=self.density)
                self.W[j, i, :] = W_.A
                M = self.sigma * np.random.randn(numInputs) # dense
                for t in range(trainLen+initLen):
                    self.p[j, i, :, t] = M
                

        # for each pattern, create numExamples instances of the noise variable
        z_noise = np.random.randn(numClasses, numPatterns, numExamples, numInputs, trainLen + initLen)
        
        # create matrix of input vectors
        data = np.zeros((numClasses, numPatterns, numExamples, numInputs, trainLen + initLen))
        self.classes = []

        for j in range(numClasses):
            for i in range(numPatterns):
                for k in range(numExamples):
                    data[j, i, k, :, :] = self.W[j, i, :, :] @ z_noise[j, i, k, :, :] + self.p[j, i, :, :] 
                    self.classes.append(j)
        # flatten the data
        self.data = data.reshape(numPatterns * numClasses * numExamples, numInputs, trainLen + initLen)
        return


class TempStruct():
    ''' 
    Adapted from https://github.com/MatthieuGilson/covariance_perceptron
    '''
    def __init__(self, numClasses, numPatterns, numExamples, numInputs, trainLen, initLen, density=0.3):
        self.numClasses = numClasses
        self.numPatterns= numPatterns
        self.numExamples = numExamples
        self.numInputs = numInputs
        self.trainLen = trainLen
        self.initLen = initLen
        self.density = density # W matrix
        self.W_pat = np.zeros([numClasses, numPatterns, numInputs, numInputs])
        for cc in range(numClasses):
            for i_pat in range(numPatterns):
                # generate antisymmetric matrix
                antisym_W = np.zeros([numInputs, numInputs])
                for i in range(numInputs):
                    for j in range(i):
                        if np.random.rand() < self.density:
                            antisym_W[j, i] = (0.5 + 0.5 * np.random.rand()) * (1 - 2 * np.random.randint(2))
                            antisym_W[i, j] = -antisym_W[j, i]
                # input mixing matrix W to obtained spatially uncorrelated inputs
                self.W_pat[cc, i_pat, :, :] = expm(-np.eye(numInputs) / 2 + antisym_W)

        # for each pattern, create numExamples instances of the noise variable
        noise_x = np.random.randn(numClasses, numPatterns, numExamples, numInputs, trainLen + initLen)
        x_tmp = np.copy(noise_x)
        # create matrix of input vectors
        data = np.zeros((numClasses, numPatterns, numExamples, numInputs, trainLen + initLen))
        self.classes = []
        for j in range(numClasses):
            for i in range(numPatterns):
                for k in range(numExamples):
                    x_tmp[j, i, k, :, 1:] += np.dot(self.W_pat[j,i,:,:], x_tmp[j, i, k, :, :-1])
                    data[j, i, k, :, :] = x_tmp[j, i, k, :, :]
                    self.classes.append(j)

        # flatten the data
        self.data = data.reshape(numPatterns * numClasses * numExamples, numInputs, trainLen + initLen)
        return
