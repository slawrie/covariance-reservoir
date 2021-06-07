''' This file contains auxiliary functions
'''

import os
import time
import numpy as np
from scipy.io import arff
import pandas as pd


def my_covariance(x):
    T = x.shape[2]
    m1 = x - x.sum(2, keepdims=1) / T
    out = np.einsum('ijk,ilk->ijl', m1, m1) / (T - 1)
    return out

def my_lag_covariance(x):
    T = x.shape[2]
    m1 = (x - x.sum(2, keepdims=1) / (T-1))[:, :, :-1]
    m2 = (x - x.sum(2, keepdims=1)/(T-1))[:, :, 1:]
    out = np.einsum('ijk,ilk->ijl', m1, m2) / (T - 2)
    return out

##############################################
### REAL DATA PREPROCESSING #######################
##############################################
# load digits data and zero pad it

def preprocess_train_data(pp, d=50):
    data = arff.loadarff(open(pp))
    df = pd.DataFrame(data[0])  # (examples, [MFCCcoefficient, class])

    # parse data into numpy array
    n_coeff = 13  # attributes
    n_examples = data[0].shape[0]
    parsed_data_train = np.zeros((n_examples, n_coeff, d))
    classes = []

    for ex in range(n_examples):
        classes.append(int(data[0]['class'][ex]))
        for t in range(1, d+1):
            name = 'coeffficient%i' % t
            for s in range(n_coeff):
                value = data[0]['MFCCcoefficient'][ex][name][s] #s
                if str(value) == 'nan':
                    parsed_data_train[ex, s, t - 1] = 0
                else:
                    parsed_data_train[ex, s, t - 1] = value

    classes = np.asarray(classes) - 1
    # save data for future use
    #np.save(pp / 'dataset' / 'train_data.npy', parsed_data_train)
    #np.save(pp / 'dataset' / 'train_labels.npy', classes)
    return parsed_data_train, classes


def preprocess_test_data(pp, d=50):
    data = arff.loadarff(open(pp))
    df = pd.DataFrame(data[0])  # (examples, [MFCCcoefficient, class])

    # parse data into numpy array
    n_coeff = 13  # attributes
    n_examples = data[0].shape[0]
    parsed_data_test = np.zeros((n_examples, n_coeff, d))
    classes = []

    for ex in range(n_examples):
        classes.append(int(data[0]['class'][ex]))
        for t in range(1, d+1):
            name = 'coefficient%i' % t
            for s in range(n_coeff):
                value = data[0]['MFCCcoefficient'][ex][name][s] #s
                if str(value) == 'nan':
                    parsed_data_test[ex, s, t - 1] = 0
                else:
                    parsed_data_test[ex, s, t - 1] = value

    classes = np.asarray(classes) - 1
    # save data for future use
    #np.save(pp / 'dataset' / 'test_data.npy', parsed_data_test)
    #np.save(pp / 'dataset' / 'test_labels.npy', classes)
    return parsed_data_test, classes

