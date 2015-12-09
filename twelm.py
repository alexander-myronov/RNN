"""
Implementation assumes that there are TWO LABELS, namely -1 and +1.
If you have different labels you have to preproess them. Furthermore
be sure to correctly set "balanced" hyperparameter accordingly to the
metric you want to optimize
"""

from __future__ import division
import numpy as np
from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator


def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX + WW - XW)

def kulczynski2(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return 0.5*XW *(1.0/XX + 1.0/WW)


metric = {
    'tanimoto': tanimoto,
    'kulczynski2': tanimoto,

}


class ELM(object):
    """ Extreme Learning Machine """

    def __init__(self, h, C=10000, f='tanimoto', random_state=666, balanced=False):
        """
        h - number of hidden units
        C - regularization strength (L2 norm)
        f - activation function [default: tanimoto]
        balanced - if set to true, model with maximize GMean (or Balanced accuracy), 
                   if set to false [default] - model will maximize Accuracy
        """
        self.h = h
        self.C = C
        self.f = metric[f]
        self.metric_name = f
        self.rs = random_state
        self.balanced = balanced

    def _hidden_init(self, X, y, hidden_layer=None):
        """ Initializes hidden layer """
        np.random.seed(self.rs)
        if hidden_layer is not None:
            W = csr_matrix(hidden_layer[:self.h])
        else:
            W = csr_matrix(np.random.rand(self.h, X.shape[1]))

        b = np.random.normal(size=self.h)
        return W, b

    def fit(self, X, y, hidden_layer=None):
        """ Fits ELM to training samples X and labels y """
        self.W, self.b = self._hidden_init(X, y, hidden_layer)
        H = self.f(X, self.W, self.b)

        if self.balanced:
            counts = {l: float(y.tolist().count(l)) for l in set(y)}
            ms = max([counts[k] for k in counts])
            self.counts = {l: np.sqrt(ms / counts[l]) for l in counts}
        else:
            self.counts = {l: 1 for l in set(y)}

        w = np.array([[self.counts[a] for a in y]]).T
        H = np.multiply(H, w)
        y = np.multiply(y.reshape(-1, 1), w).ravel()

        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        return np.array(np.sign(H.dot(self.beta)).tolist())

    def get_params(self, deep=True):
        return {'h': self.h, 'C': self.C, 'f': self.metric_name}

    def set_params(self, **params):
        self.h = params['h']
        self.C = params['C']
        if 'f' in params:
            self.metric_name = params['f']
            self.f = metric[self.metric_name]
        return self


class XELM(ELM):
    """ Extreme Learning Machine initialized with training samples """

    def _hidden_init(self, X, y, hidden_layer=None):
        h = min(self.h, X.shape[0])  # hidden neurons count can't exceed training set size

        np.random.seed(self.rs)
        if hidden_layer is not None:
            print(X.shape)
            raise NotImplementedError()
            W = hidd
        else:
            W = X[np.random.choice(range(X.shape[0]), size=h, replace=False)]
        b = np.random.normal(size=h)
        return W, b


class TWELM(XELM):
    """ 
    TWELM* model from

    "Weighted Tanimoto Extreme Learning Machine with case study of Drug Discovery"
    WM Czarnecki, IEEE Computational Intelligence Magazine, 2015
    """

    def __init__(self, h, C=10000, random_state=None):
        super(self.__class__, self).__init__(h=h, C=C, f='tanimoto', random_state=random_state,
                                             balanced=True)


def euclid(X, W, b=None):
    d = cdist(X, W, metric='cityblock')
    return d


class ELMRegressor(XELM):
    def __init__(self, h, f, C=10000, random_state=666):
        super(self.__class__, self).__init__(h=h, C=C, f=f, random_state=random_state,
                                             balanced=False)

    def fit(self, X, y, hidden_layer=None):
        """ Fits ELM to training samples X and labels y """
        self.W, self.b = self._hidden_init(X, y, hidden_layer)
        H = self.f(X, self.W, self.b)
        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        return H.dot(self.beta)


class RBFNet(XELM):
    def __init__(self, h, f='tanimoto', C=10000, b=1, random_state=None):
        super(self.__class__, self).__init__(h, C, f, random_state, balanced=False)
        self.b = b  # np.random.uniform(0.4, 2.4, size=h)

    def rbf(self, dist):
        return np.exp(np.power(np.array(dist), 2) * (-self.b))

    def fit(self, X, y, hidden_layer=None):
        """ Fits ELM to training samples X and labels y """
        self.W, self.b = self._hidden_init(X, y, hidden_layer)
        H = self.f(X, self.W, self.b)
        H = self.rbf(H)

        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        H = self.rbf(H)
        return np.array(np.sign(H.dot(self.beta)).tolist())

    def get_params(self, deep=True):
        params = super(self.__class__, self).get_params(deep)
        params['b'] = self.b
        return params

    def set_params(self, **params):
        super(self.__class__, self).set_params(**params)
        self.b = params['b']
        return self
