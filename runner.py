# -*- coding: utf-8 -*-
# In[ ]:
import argparse
import cProfile
from collections import OrderedDict
from io import StringIO
import json
import os
import pstats
import traceback
import datetime
from ipyparallel import CompositeError
import ipyparallel

import numpy as np
import itertools
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import scipy
from six import BytesIO
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from kfold_repeat import KFoldRepeat
from loader import get_datasets_with_dud, loader
from parallel_grid_search import GridSearchCVParallel
from twelm import TWELM, XELM, RBFNet, EEM, ELM

from sklearn.metrics import confusion_matrix


# In[ ]:



# In[ ]:

import sys
# from twelm_theano import XELMTheano
# from twelm_theano import XELMTheano
#



# In[ ]:
from twelm_theano import XELMTheano, RBFNetTheano, EEMTheano

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
    import numpy as np
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
                        n_inner_folds, n_outer_repetitions, n_inner_repetitions,
                        X_name='X', y_name='y'):
    """
    returns
    test score for each outer fold
    best score from grid search
    best estimator parameters for each iteration
    """
    test_scores = np.zeros(n_outer_folds * n_outer_repetitions)
    train_scores = np.zeros(n_outer_folds * n_outer_repetitions)
    best_parameters = []

    if issparse(features):
        features = features.toarray()

    for rep in range(n_outer_repetitions):
        print('%d/%d repetition' % (rep + 1, n_outer_repetitions))
        fold = StratifiedKFold(activity, n_folds=n_outer_folds, shuffle=True)
        # print(len(fold))
        fit_params = {}
        if isinstance(estimator, XELM) and 'h' in param_grid:
            max_h = min(features.shape[0], max(param_grid['h']))
            fit_params['hidden_layer'] = features[np.random.choice(features.shape[0],
                                                                   max_h,
                                                                   replace=False)]
        for i, (train_index, test_index) in enumerate(fold):
            print('%d/%d fold' % (i + 1, n_outer_folds))

            def gs_callback(gs_progress, gs_length, elapsed):
                print('[%s]: cv: %d/%d, grid search: %d/%d' % \
                      (elapsed,
                       rep * n_outer_folds + i + 1,
                       n_outer_repetitions * n_outer_folds,
                       gs_progress,
                       gs_length))

            view = None
            if 'view' in globals():
                view = globals()['view']
            search = GridSearchCV(estimator, param_grid, scoring=scorer,
                                          cv=KFoldRepeat(activity[train_index],
                                                         n_folds=n_inner_folds,
                                                         n_reps=n_inner_repetitions),
                                          fit_params=fit_params,
                                          verbose=0,
                                          #view=view,
                                          #callback=gs_callback,
                                          refit=False)

            search.fit(features[train_index], activity[train_index])
                       #x_is_index=True,
                       #X_name=X_name,
                       #y_name=y_name)

            best_estimator = clone(estimator).set_params(**search.best_params_)
            # best_estimator = search.best_estimator_
            best_estimator.fit(features[train_index], activity[train_index])

            test_score = scorer(best_estimator, features[test_index], activity[test_index])
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
    name = name.replace('RBFNetTheano', 'RBFNet')  # dirty hack
    return name


# In[ ]:

def test_models(estimators, estimator_grids, X, Y, scorer, n_outer_folds, n_inner_folds,
                n_outer_repetitions, n_inner_repetitions, X_name='X', y_name='y'):
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
                                                n_outer_repetitions=n_outer_repetitions,
                                                n_inner_repetitions=n_inner_repetitions,
                                                X_name=X_name,
                                                y_name=y_name)
        print(scores_test)
        estimator_scores.append(scores_test)

    return estimator_scores


# In[ ]:

# from twelm_theano import XELMTheano, EEMTheano, RBFNetTheano

estimators = [
    EEMTheano(h=100, f='tanimoto'),
    EEMTheano(h=100, f='kulczynski2'),
    EEMTheano(h=100, f='kulczynski3'),
    EEMTheano(h=100, f='f1_score'),
    RBFNetTheano(h=100),
    XELMTheano(h=10, f='tanimoto', balanced=True),
    XELMTheano(h=10, f='kulczynski2', balanced=True),
    XELMTheano(h=10, f='kulczynski3', balanced=True),
    XELMTheano(h=10, f='f1_score', balanced=True),
    RandomForestClassifier(n_jobs=-1),
    SVC(),
    LogisticRegression(n_jobs=-1),
]

+estimator_grids = [
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 100000], 'h': [200, 400, 800, 1000], 'b': [0.01, 0.1, 1]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    # {'C': [1000, 100000], 'h': [500, 1500]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {'C': [100, 1000, 10000], 'h': [200, 400, 800, 1000]},
    {
        'n_estimators': [25, 75, 125],
        'max_depth': [3, 5, 7, None],
        'min_samples_leaf': [1, 5, 10],

    },
    {
        'C': [1, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': [ 1, 0.01, 1e-7, 'auto'],
        'degree': [2, 3, 4]
    },
    {
        'C': [0.1, 1, 10, 100, 1000],
    },

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
    # XELMTheano(f='euclidean', balanced=True),
    # EEMTheano(f='euclidean'),
    # RBFNetTheano(),
    RandomForestClassifier(),
    SVC(),
    LogisticRegression(solver='liblinear'),
]

estimator_grids_toy = [
    # {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    # {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000]},
    # {'C': [1, 10, 100, 1000, 10000], 'h': [100, 200, 300, 400, 800, 1000], 'b': [0.1, 0.5, 1]},
    {
        'n_estimators': [20, 40, 60, 100],
        'max_depth': [4, 5, 6, 7],
        'max_features': ['sqrt', 'log2', None]
    },
    {'kernel': ['linear', 'rbf', 'poly'], 'C': [1, 10, 100, 1000, 10000]},
    {'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000, 10000]},

]


def prepare_and_train(name, X, y, estimator_index, n_inner_folds=3, n_inner_repetitions=1,
                      n_outer_folds=3, n_outer_repetitions=1):
    print(name, get_estimator_descritpion(estimators[estimator_index]))

    try:
        # pr = cProfile.Profile()
        # pr.enable()

        X_filename = r'data/cache/' + name + '_X'
        y_filename = r'data/cache/' + name + '_y'
        estimators_scores = test_models([estimators[estimator_index]],
                                        [estimator_grids[estimator_index]],
                                        X,
                                        y,
                                        scorer=bac_scorer,
                                        n_outer_folds=n_outer_folds,
                                        n_outer_repetitions=n_outer_repetitions,
                                        n_inner_folds=n_inner_folds,
                                        n_inner_repetitions=n_inner_repetitions,
                                        X_name=X_filename,
                                        y_name=y_filename)
        # pr.disable()

        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # ps.print_stats()
    except Exception:
        e_type, e_value, e_tb = sys.exc_info()
        # print(e_tb)

        return name, estimator_index, (e_value, e_tb)
    print('%s ready' % name)
    return name, estimator_index, estimators_scores


def setup_ipp(remote):
    import ipyparallel as ipp
    if remote:
        client = ipp.Client(profile_dir='/shared/ipython/profile_ssh')
        # ipp.use_dill()
        client[:].map(os.chdir, ['/shared/RNN'] * len(client[:]), block=True)
        print(client[:].apply_sync(os.getcwd))
    else:
        client = ipp.Client()

    view = client.load_balanced_view()
    return view


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='cross validation number of folds')
    parser.add_argument('--cv_reps', type=int, default=1,
                        help='cross validation number of repetitions')
    parser.add_argument('--gs_folds', type=int, default=3,
                        help='grid search number of folds')
    parser.add_argument('--gs_reps', type=int, default=1,
                        help='grid search number of repetitions')
    parser.add_argument('--file', type=str,
                        default=r'data/results%s.csv' % \
                                str(datetime.datetime.now()).replace(':', '.'),
                        help='results file name')

    args = parser.parse_args()

    cache = True

    if cache:
        with open(r'data/cache/filenames.json', 'r') as f:
            filenames = json.load(f, object_pairs_hook=OrderedDict)

        datasets = OrderedDict()
        for name, (X_filename, y_filename) in filenames.iteritems():
            # if 'DUD' not in name:
            #     continue

            datasets[name] = loader(X_filename, y_filename)

    else:
        datasets = get_datasets_with_dud(datafiles)
        filenames = OrderedDict()
        for name, (X, y) in datasets.iteritems():
            X_filename = r'data/cache/' + name + '_X'
            y_filename = r'data/cache/' + name + '_y'

            if issparse(X):
                X = X.toarray()
            if issparse(y):
                y = y.toarray()

            np.save(X_filename, X)
            np.save(y_filename, y)
            filenames[name] = (X_filename, y_filename)

        with open(r'data/cache/filenames.json', 'w') as f:
            json.dump(filenames, f, indent=4)

    # exit()

    estimators = estimators
    estimator_grids = estimator_grids

    filename = r'data/results_debug.csv'
    filename = args.file

    #view = setup_ipp(remote=True)
    #globals()['view'] = view
    # view.client[:].push(
    #     {
    #         'bac_scorer': bac_scorer,
    #         'bac_error': bac_error,
    #         'process_cm': process_cm,
    #         'confusion_matrix':confusion_matrix,
    #         #'np': np,
    #
    #     })


    # def imports():
    #
    #
    #
    # def check_imports():
    #     try:
    #         print(bac_scorer)
    #         # print(confusion_matrix)
    #         return True
    #     except:
    #         return False


    # view.client[:].apply_sync(imports)
    #
    # print(view.client[:].apply_sync(check_imports))

    if os.path.isfile(filename):
        scores_grid = pd.read_csv(filename)
        scores_grid.loc[:, 'dataset'] = pd.Series(data=datasets.keys())
    else:
        scores_grid = pd.DataFrame(dtype=object)
        scores_grid.loc[:, 'dataset'] = pd.Series(data=datasets.keys())

    scores_grid.set_index('dataset', inplace=True, drop=False)
    # print(datasets.keys())
    # print(scores_grid.columns)
    # print(scores_grid.index)
    scores_grid = scores_grid.reindex(pd.Series(data=datasets.keys()))

    for estimator in estimators:
        column_name = get_estimator_descritpion(estimator)
        if column_name not in scores_grid.columns:
            scores_grid.loc[:, column_name] = pd.Series(
                [''] * len(datasets)).astype('str')
        scores_grid.loc[:, column_name] = scores_grid.loc[:, column_name].astype('str')


    def map_callback(result):
        dataset_name, estimator_index, estimators_scores = result
        estimator_name = get_estimator_descritpion(estimators[estimator_index])
        print('callback %s dataset %s model' % (dataset_name, estimator_name))
        #  print(result[2])

        if isinstance(estimators_scores[0], Exception):

            tb = ''.join(traceback.format_tb(estimators_scores[1]))
            print(
                'callback: datafile=%s, model=%s exception=%s \n traceback\n %s' % (
                    dataset_name, estimator_name, repr(estimators_scores[0]), tb))
            if isinstance(estimators_scores[0], CompositeError):
                estimators_scores[0].print_traceback()
        else:
            scores_grid.set_value(dataset_name, estimator_name, str(estimators_scores[0]))
            scores_grid.to_csv(filename, index=False)


    start_time = time.time()

    parallel_models = False
    if parallel_models:

        import multiprocessing as mp

        mp.freeze_support()
        pool = mp.Pool(processes=4)

        async_results = []
        for (name, (dataset)), estimator_index in itertools.product(datasets.items()[::1],
                                                                    xrange(0, len(estimators))):
            value = \
                str(scores_grid.loc[name, get_estimator_descritpion(estimators[estimator_index])])

            try:
                value = np.fromstring(value.replace('[', '').replace(']', ''), sep=' ')
                if np.isnan(value).any():
                    raise Exception
            except:
                value = None

            if callable(dataset):
                X, y = dataset()
            else:
                X, y = dataset

            if value is None or len(value) < args.cv_folds * args.cv_reps:
                res = pool.apply_async(prepare_and_train,
                                       args=(name,
                                             X,
                                             y,
                                             estimator_index,
                                             args.gs_folds,
                                             args.gs_reps,
                                             args.cv_folds,
                                             args.cv_reps),
                                       callback=map_callback)
                async_results.append(res)
        while not all(map(lambda r: r.ready(), async_results)):
            pass
        # result_series = map(lambda ar: ar.get(), async_results)
        pool.close()
    else:

        #

        for (name, (dataset)), estimator_index in itertools.product(datasets.items()[::1],
                                                                    xrange(0, len(estimators))):
            value = \
                str(scores_grid.loc[name, get_estimator_descritpion(estimators[estimator_index])])

            try:
                value = np.fromstring(value.replace('[', '').replace(']', ''), sep=' ')
                if np.isnan(value).any():
                    raise Exception
            except:
                value = None

            if callable(dataset):
                X, y = dataset()
            else:
                X, y = dataset

            if value is None or len(value) < args.cv_folds * args.cv_reps:
                map_callback(prepare_and_train(name, X, y, estimator_index,
                                               n_inner_folds=args.gs_folds,
                                               n_inner_repetitions=args.gs_reps,
                                               n_outer_folds=args.cv_folds,
                                               n_outer_repetitions=args.cv_reps,
                                               ))

    end_time = time.time()

    print 'training done in %2.2f sec' % (end_time - start_time)

    exit()
