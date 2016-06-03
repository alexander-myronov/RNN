# -*- coding: utf-8 -*-
# In[ ]:
import cProfile
from io import StringIO
import os
import pstats

import numpy as np
import itertools
import pandas as pd
from six import BytesIO
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from twelm import TWELM, XELM, RBFNet, EEM, ELM

from sklearn.metrics import confusion_matrix


# In[ ]:



# In[ ]:

import sys
# from twelm_theano import XELMTheano
# from twelm_theano import XELMTheano
from twelm_theano import XELMTheano, EEMTheano, RBFNetTheano

oldsysstdout = sys.stdout


class flushfile():
    def __init__(self, f):
        self.f = f

    def __getattr__(self, name):
        return object.__getattribute__(self.f, name)

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()


sys.stdout = flushfile(sys.stdout)


# In[ ]:

datafiles = [
    r'data/5ht2a_ExtFP.libsvm',
    r'data/5ht2c_ExtFP.libsvm',
    r'data/5ht6_ExtFP.libsvm',
    r'data/5ht7_ExtFP.libsvm',
    r'data/M1_ExtFP.libsvm',
    r'data/SERT_ExtFP.libsvm',
    r'data/cathepsin_ExtFP.libsvm',
    r'data/d2_ExtFP.libsvm',
    r'data/h1_ExtFP.libsvm',
    r'data/hERG_ExtFP.libsvm',
    r'data/hiv_integrase_ExtFP.libsvm',
    r'data/hiv_protease_ExtFP.libsvm',
]

datafiles_toy = [
    r'data/diabetes_scale.libsvm',
    r'data/australian_scale.libsvm',
    r'data/breast-cancer_scale.libsvm',
    r'data/german.numer_scale.libsvm',
    r'data/ionosphere_scale.libsvm',
    r'data/liver-disorders_scale.libsvm',
    r'data/heart_scale.libsvm',
    r'data/sonar_scale.libsvm',
]


# In[ ]:

def process_cm(confusion_mat, i=0):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i, i]  # correctly labeled as i
    FP = confusion_mat[:, i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i, :].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    return TP, FP, FN, TN


# In[ ]:

def bac_error(Y, Y_predict):
    cm = confusion_matrix(Y, Y_predict)
    bac_values = np.zeros(cm.shape[0])
    for i in xrange(cm.shape[0]):
        tp, fp, fn, tn = process_cm(cm, i=i)
        if tp + fn > 0 and tn + fp > 0:
            bac_values[i] = 0.5 * tp / (tp + fn) + 0.5 * tn / (tn + fp)
    return bac_values


# In[ ]:

def bac_scorer(estimator, X, Y):
    Y_predict = estimator.predict(X)
    bac_values = bac_error(Y, Y_predict)
    return np.mean(bac_values)


# In[ ]:

def perform_grid_search(estimator, features, activity, scorer, param_grid, n_outer_folds,
                        n_inner_folds, n_outer_repetitions):
    """
    returns
    test score for each outer fold
    best score from grid search
    best estimator parameters for each iteration
    """
    test_scores = np.zeros(n_outer_folds * n_outer_repetitions)
    train_scores = np.zeros(n_outer_folds * n_outer_repetitions)
    best_parameters = []
    for rep in range(n_outer_repetitions):
        # print('%d/%d' % (rep, n_outer_repetitions))
        fold = StratifiedKFold(activity, n_folds=n_outer_folds, shuffle=True)
        # print(len(fold))
        fit_params = {}
        if isinstance(estimator, XELM) and 'h' in param_grid:
            max_h = min(features.shape[0], max(param_grid['h']))
            fit_params['hidden_layer'] = features[np.random.choice(features.shape[0],
                                                                   max_h,
                                                                   replace=False)]
        for i, (train_index, test_index) in enumerate(fold):
            search = GridSearchCV(estimator, param_grid, scoring=scorer,
                                  cv=n_inner_folds,
                                  n_jobs=1,
                                  fit_params=fit_params)

            search.fit(features[train_index], activity[train_index])

            estimator_train = clone(estimator)
            estimator_train.set_params(**search.best_params_)
            estimator_train.fit(features[train_index], activity[train_index])

            test_score = scorer(estimator_train, features[test_index], activity[test_index])
            test_scores[rep * n_outer_folds + i] = test_score
            train_scores[rep * n_outer_folds + i] = search.best_score_

            best_parameters.append(search.best_params_)
            #             print('train score=%f, test score=%f' % (search.best_score_, test_score))
            print(search.best_params_)
    return test_scores, train_scores, best_parameters


# In[ ]:

def get_estimator_descritpion(estimator):
    name = type(estimator).__name__
    if hasattr(estimator, 'metric_name') and not isinstance(estimator, RBFNet):
        name += '(%s)' % estimator.metric_name
    return name


# In[ ]:

def test_models(estimators, estimator_grids, X, Y, scorer, n_outer_folds, n_inner_folds,
                n_outer_repetitions):
    estimator_scores = []
    # estimator_scores_std = np.zeros(len(estimators))
    assert len(estimators) <= len(estimator_grids)
    for i, (estimator, grid) in enumerate(itertools.izip(estimators, estimator_grids)):
        print(get_estimator_descritpion(estimator))

        scores_test, _, _ = perform_grid_search(estimator,
                                                X,
                                                Y,
                                                param_grid=grid,
                                                scorer=scorer,
                                                n_outer_folds=n_outer_folds,
                                                n_inner_folds=n_inner_folds,
                                                n_outer_repetitions=n_outer_repetitions)
        print(scores_test)
        estimator_scores.append(scores_test)

    return estimator_scores


# In[ ]:

estimators = [
    EEMTheano(h=100, f='tanimoto'),
    EEMTheano(h=100, f='kulczynski2'),
    EEMTheano(h=100, f='kulczynski3'),
    EEMTheano(h=100, f='f1_score'),
    RBFNet(h=100),
    XELMTheano(h=10, f='tanimoto', balanced=True),
    XELMTheano(h=10, f='kulczynski2', balanced=True),
    XELMTheano(h=10, f='kulczynski3', balanced=True),
    XELMTheano(h=10, f='f1_score', balanced=True),
    # RandomForestClassifier(n_jobs=-1)
]

estimator_grids = [
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [1000, 100000], 'h': [500, 1500], 'b': [0.4, 1, 2]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'n_estimators': [25, 75, 125]}
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
]

estimator_grids_simple = [
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [None], 'h': [50]},
    {'C': [1000], 'h': [50], 'b': [0.4]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'C': [1000], 'h': [50]},
    {'n_estimators': [7, 12]}

]

estimators_toy = [
    XELMTheano(f='euclidean', balanced=True),
    EEMTheano(f='euclidean'),
    RBFNetTheano(),
    RandomForestClassifier(),
    SVC(),
    LogisticRegression(solver='liblinear'),
]

estimator_grids_toy = [
    {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000], 'b': [0.1, 0.5, 1]},
    {
        'n_estimators': [20, 40, 60, 100],
        'max_depth': [4, 5, 6, 7],
        'max_features': ['sqrt', 'log2', None]
    },
    {'kernel': ['linear', 'rbf', 'poly'], 'C': [1, 10, 100, 1000, 10000]},
    {'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000, 10000]},

]


def prepare_and_train(datafile):
    print(datafile)
    features, activity = load_svmlight_file(datafile)
    s = BytesIO()
    try:
        # pr = cProfile.Profile()
        # pr.enable()
        estimators_scores = test_models(estimators_toy,
                                        estimator_grids_toy,
                                        features,
                                        activity,
                                        scorer=bac_scorer,
                                        n_outer_folds=3,
                                        n_inner_folds=3,
                                        n_outer_repetitions=5)
        # pr.disable()

        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # ps.print_stats()
    except Exception as e:
        print(e)
        return datafile, e, None
    print('%s ready' % datafile)
    return datafile, estimators_scores, s.getvalue()


if __name__ == '__main__':

    # import theano
    # theano.config.exception_verbosity ='high'

    import multiprocessing as mp

    mp.freeze_support()

    datafiles = datafiles_toy
    estimators = estimators_toy
    estimator_grids = estimator_grids_toy

    filename = r'data/results_toy.csv'

    if os.path.isfile(filename):
        scores_grid = pd.read_csv(filename)
    else:
        scores_grid = pd.DataFrame(dtype=object)
        scores_grid.loc[:, 'dataset'] = pd.Series(data=datafiles)

    scores_grid.set_index('dataset', inplace=True, drop=False)
    scores_grid = scores_grid.reindex(pd.Series(data=datafiles))

    for estimator in estimators:
        column_name = get_estimator_descritpion(estimator)
        if column_name not in scores_grid.columns:
            scores_grid.loc[:, column_name] = pd.Series(
                [''] * len(datafiles)).astype('str')
            scores_grid.loc[:, column_name] = scores_grid.loc[:, column_name].astype('str')

    pool = mp.Pool(1)


    def map_callback(result):
        print('done %s dataset' % result[0])
        print(result[2])
        datafile, estimators_scores, _ = result
        if estimators_scores is Exception:
            print('callback: datafile=%s, exception=%s' % (datafile, estimators_scores))
        for estimator, score in itertools.izip(estimators, estimators_scores):
            scores_grid.set_value(datafile, get_estimator_descritpion(estimator), str(score))
        scores_grid.to_csv(filename, index=False)


    start_time = time.time()

    async_results = [pool.apply_async(prepare_and_train, args=(datafile,), callback=map_callback)
                     for datafile in datafiles[:] if
                     any(map(lambda v: pd.isnull(v) or v == 'nan', scores_grid.loc[datafile]))]

    while not all(map(lambda r: r.ready(), async_results)):
        pass

    result_series = map(lambda ar: ar.get(), async_results)

    pool.close()

    end_time = time.time()

    print 'training done in %2.2f sec' % (end_time - start_time)

    exit()

    # In[ ]:



    # scores_values = pd.DataFrame()
    # scores_values.loc[:, 'dataset'] = pd.Series()
    # for estimator in estimators:
    #     scores_values.loc[:, get_estimator_descritpion(estimator)] = pd.Series(index=np.arange(len(datafiles)))

    scores_values = {}

    for i, (datafile, estimators_scores) in enumerate(result_series):
        print(datafile)

        scores_grid.ix[i, 'dataset'] = datafile
        # scores_values.ix[i, 'dataset'] = datafile



        for estimator, score in itertools.izip(estimators, estimators_scores):
            scores_grid.ix[i, get_estimator_descritpion(estimator) + '_score'] = score.mean()
            scores_grid.ix[i, get_estimator_descritpion(estimator) + '_std'] = score.std()
            # scores_values.ix[i, get_estimator_descritpion(estimator)] = score
            scores_values[(datafile, get_estimator_descritpion(estimator))] = score


    # In[ ]:

    import cPickle

    with open(r'data/scores.pkl', 'wb') as f:
        cPickle.dump(scores_values, f, -1)


    # In[ ]:

    scores_grid_ds = scores_grid.set_index('dataset')


    # In[ ]:

    import scipy.stats


    # In[ ]:

    for (datafile, estimator_name), scores in scores_values.iteritems():
        scores_grid_ds.ix[datafile, (estimator_name + '_stats')] = scipy.stats.shapiro(scores)[1]


    # In[ ]:

    # scores_grid_ds


    # In[ ]:

    scores_grid_ds.to_csv(r'data/scores_grid_ds.csv', index=False)


# In[ ]:
