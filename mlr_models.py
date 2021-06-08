import numpy as np
import auxiliary_functions as af
import time
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle

def mlr_decoder(alpha):
    model = make_pipeline(StandardScaler(), \
                          LogisticRegression(C=alpha, solver='lbfgs', fit_intercept=True,
                                             multi_class='multinomial', random_state=0, max_iter=500))

    return model

def get_regularization_parameter(X, y, alphas):
    # for each alpha, do 5-fold cv, stratified classes
    dic = {'alphas': [], 'train_scores': [], 'test_scores': []}

    for alpha in alphas:
        model = mlr_decoder(alpha)
        train_scores = []
        test_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        for train_index, test_index in skf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            model.fit(X_train, y_train)
            train_scores.append(model.score(X_train, y_train))
            test_scores.append(model.score(X_test, y_test))

        dic['alphas'].append(alpha)
        dic['train_scores'].append(train_scores)
        dic['test_scores'].append(test_scores)
        print(f'Alpha={alpha} finished')

    # get best parameter (higher test error, decide ties with lower sem
    cv_scores = []
    sems = []
    for i in range(len(dic['alphas'])):
        cv_scores.append(np.mean(dic['test_scores'][i]))
        sems.append(sem(dic['test_scores'][i]))

    max_score = max(cv_scores)
    indices = [index for index, value in enumerate(cv_scores) if value == max_score]

    if len(indices) > 1:
        min_sem = min([sems[index] for index in indices])
        sem_index = [index for index, value in enumerate(sems) if value == min_sem]
        best_alpha = dic['alphas'][sem_index[-1]]
    else:
        best_alpha = dic['alphas'][indices[0]]
    return dic, best_alpha

