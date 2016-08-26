from collections import OrderedDict
import os
import numpy as np
import scipy
from scipy.sparse import issparse, csr_matrix
from sklearn.datasets import load_svmlight_file
import pandas as pd

__author__ = 'Alex'


def loader(X_name, y_name):
    return lambda: (np.load(X_name + '.npy', mmap_mode='r'), \
                    np.load(y_name + '.npy', mmap_mode='r'))


def get_datasets(datafiles):
    result = {}
    for filename in datafiles:
        result[os.path.basename(filename).replace('.libsvm', '')] = load_svmlight_file(filename)
    return result


def get_dud_filename(filename):
    filename_wo_extension = filename.replace('.libsvm', '')
    parts = filename_wo_extension.split('_')
    return 'data/' + parts[0] + '_DUD_' + parts[1] + '.csv'


def load_dud(filename):
    df = pd.read_csv(filename)
    columns = list(df.columns)
    columns.remove('Name')
    values = df.loc[:, columns].values

    return values, np.full(len(values), fill_value=-1)


def get_datasets_with_dud(datafiles):
    result = OrderedDict()
    for filename in datafiles:
        X, y = load_svmlight_file(filename)
        # if issparse(X): #TODO: sparse option
        #     X = X.toarray()
        filename = os.path.basename(filename).replace('.libsvm', '')
        result[filename] = (X, y)

        dud_filename = get_dud_filename(filename)
        if not os.path.isfile(dud_filename):
            continue
        X_dud, y_dud = load_dud(dud_filename)
        X_dud = csr_matrix(X_dud)

        pad_length = X_dud.shape[1] - X.shape[1]

        if issparse(X):
            X = scipy.sparse.hstack([X, csr_matrix(np.zeros(shape=(X.shape[0], pad_length)))],
                                    format='csr')
        else:
            X = np.hstack([X, np.zeros(shape=(X.shape[0], pad_length))])

        for percent in [0.1, 0.5, 1]:
            indices = np.random.choice(X_dud.shape[0], int(X_dud.shape[0] * percent), replace=False)
            X_mixin = X_dud[indices]
            y_mixin = y_dud[indices]

            result['%s+%d%%DUD' % (filename, int(percent * 100))] = \
                (scipy.sparse.vstack([X, X_mixin], format='csr'), np.concatenate([y, y_mixin]))
            del X_mixin
            X_mixin = None

        del X_dud
        X_dud = None

    return result
