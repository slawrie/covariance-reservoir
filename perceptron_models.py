import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

def my_covariance(x):
    N = x.shape[2]
    m1 = x - x.sum(2, keepdims=1) / N
    out = np.einsum('ijk,ilk->ijl', m1, m1) / (N - 1)
    return out


class Perceptron:

    def __init__(self, inSize, outSize, leak=1.0, random_state=42, non_linear=False):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.inSize = inSize
        self.outSize = outSize
        self.Wout = np.random.rand(self.outSize, self.inSize + 1) - 0.5
        self.leak = leak  # not being used now
        self.outStates = None
        self.outCovariance = None
        self.outMean = None
        self.non_linear = non_linear
        return

    def run(self, data, initLen, trainLen, covariance=False, mean=False):
        '''Data is an array. Dimension is (numExamples, numInputs, timeLen)'''

        # add bias unit to input data
        ones = np.ones((data.shape[0], 1, trainLen-initLen))
        inputs = np.concatenate((ones, data[:, :, initLen:trainLen]), axis=1)
        if self.non_linear:
            self.outStates = np.tanh(np.einsum('ij,ljk -> lik', self.Wout, inputs))
        else:
            self.outStates = np.einsum('ij,ljk -> lik', self.Wout, inputs)

        if covariance:  # update covariances
            self.outCovariance = my_covariance(self.outStates)

        if mean:  # update mean states
            self.outMean = np.mean(self.outStates, axis=2)
        return

    def predict(self, mode='mean'):
        #Run data through through perceptron, get covariances in output units. If var0 > var 1, class is 0.

        Y = []
        if mode == 'mean':
            for ex in range(self.outStates.shape[0]):
                max_out = np.max(self.outMean[ex, :])
                pred = np.where(self.outMean[ex, :] == max_out)[0][0]
                Y.append(pred)

        if mode == 'covariance':
            for ex in range(self.outStates.shape[0]):
                diagonals = np.diag(self.outCovariance[ex, :, :])
                max_out = np.max(diagonals)
                pred = np.where(diagonals == max_out)[0][0]
                Y.append(pred)
        return Y


    def score(self, Y_true, Y_pred):
        return accuracy_score(Y_true, Y_pred)

    def plot_output_units(self):
        '''Plot some random reservoir unit activity during a random input presentation'''
        if self.outStates is None:
            print('Run data to update output states!')
            return
        else:
            print('Plotting activity')
            # plot output units for a random example
            fig = plt.figure()
            time = [i for i in range(self.outStates.shape[2])]
            sample = np.random.choice(self.outStates.shape[0])
            plt.plot(time, self.outStates[sample, :, :].T)
            plt.xlabel('Steps (input %i)' % sample)
            plt.ylabel('Output activation')
            fig.savefig('outputActivity.png', dpi=200, bbox_inches='tight')
            print('Done')
            return
