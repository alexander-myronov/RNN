"""
Models and metrics
"""

from __future__ import division
import numpy as np
from scipy import linalg as la
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf


def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX + WW - XW)


def kulczynski2(X, W, b=None):
    """ Kulczynski2 similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return 0.5 * (XW / XX + XW / WW)


def f1_score(X, W, b=None):
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return np.reciprocal(XW / XX) + np.reciprocal(XW / WW)


def kulczynski3(X, W, b=None):
    """
    GMean of precision and recall
    """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return np.sqrt(np.multiply((XW / XX), (XW / WW)))


def euclidean(X, W, b=None):
    # TODO calc pairwise distance for sparse matrices
    if scipy.sparse.issparse(X):
        X = X.toarray()
    if scipy.sparse.issparse(W):
        W = W.toarray()
    d = cdist(X, W, metric='euclidean')
    return d


def gaussian(x, mu, sigma):
    return np.exp(np.power((x - mu), 2) / (-2 * sigma)) / (sigma * np.sqrt(2 * np.pi))


metric = {
    'tanimoto': tanimoto,
    'kulczynski2': kulczynski2,
    'euclidean': euclidean,
    'kulczynski3': kulczynski3,
    'f1_score': f1_score,

}


class ELM(object):
    """ Extreme Learning Machine """

    def __init__(self, f, h=1, C=10000, random_state=666, balanced=False):
        """
        h - number of hidden units
        C - regularization strength (L2 norm)
        f - activation function [default: tanimoto]
        balanced - if set to true, model with maximize GMean (or Balanced accuracy),
                   if set to false [default] - model will maximize Accuracy
        """
        self.h = h
        self.C = C
        if f in metric:
            self.f = metric[f]
            self.metric_name = f
        else:
            self.f = f
            self.metric_name = 'custom'

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

    def __getstate__(self):
        return self.get_params()

    def __setstate__(self, params):
        self.set_params(params)


class XELM(ELM):
    """ Extreme Learning Machine initialized with training samples """

    def _hidden_init(self, X, y, hidden_layer=None):
        h = min(self.h, X.shape[0])  # hidden neurons count can't exceed training set size

        np.random.seed(self.rs)
        if hidden_layer is not None:
            # print('using hidden layer')
            W = hidden_layer[:h]
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

    def __init__(self, h=1, C=10000, random_state=None):
        super(self.__class__, self).__init__(h=h, C=C, f='tanimoto', random_state=random_state,
                                             balanced=True)


class ELMRegressor(XELM):
    def __init__(self, f, h=1, C=10000, random_state=666):
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
    def __init__(self, h=1, C=10000, b=1, random_state=None):
        super(RBFNet, self).__init__(f='euclidean', h=h, C=C, random_state=random_state,
                                     balanced=False)
        self.b = b  # np.random.uniform(0.4, 2.4, size=h)

    def rbf(self, dist):
        return np.exp(np.power(np.array(dist), 2) * (-self.b))

    def fit(self, X, y, hidden_layer=None):
        """ Fits ELM to training samples X and labels y """
        self.W, _ = self._hidden_init(X, y, hidden_layer)
        H = self.f(X, self.W, self.b)
        H = self.rbf(H)

        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        H = self.rbf(H)
        return np.sign(H.dot(self.beta))

    def get_params(self, deep=True):
        return {'h': self.h, 'C': self.C, 'b': self.b}

    def set_params(self, **params):
        self.b = params['b']
        self.h = params['h']
        self.C = params['C']
        return self


class EEM(XELM):
    def __init__(self, f, h=1, C=10000, random_state=666):
        super(EEM, self).__init__(h=h, C=C, f=f, random_state=random_state,
                                  balanced=False)

    def fit(self, X, y, hidden_layer=None):
        self.W, self.b = self._hidden_init(X, y, hidden_layer)
        H = self.f(X, self.W, self.b)

        plus_indices = y == 1
        H_plus = H[plus_indices]
        H_minus = H[~plus_indices]

        self.m_plus = np.mean(H_plus, axis=0).T
        self.m_minus = np.mean(H_minus, axis=0).T

        self.sigma_plus = LedoitWolf().fit(H_plus).covariance_
        self.sigma_minus = LedoitWolf().fit(H_minus).covariance_

        if self.C is not None:
            self.sigma_plus += (np.eye(len(self.sigma_plus)) / 2 * self.C)
            self.sigma_minus += (np.eye(len(self.sigma_minus)) / 2 * self.C)

        mean_diff = self.m_plus - self.m_minus
        self.beta = 2 * la.inv(self.sigma_plus + self.sigma_minus) * mean_diff / la.norm(mean_diff)
        pass

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        x = H.dot(self.beta)
        r_plus = gaussian(x, np.dot(self.beta.T, self.m_plus),
                          np.dot(np.dot(self.beta.T, self.sigma_plus), self.beta))
        r_minus = gaussian(x, np.dot(self.beta.T, self.m_minus),
                           np.dot(np.dot(self.beta.T, self.sigma_minus), self.beta))
        result = np.argmax(np.hstack([r_minus, r_plus]), axis=1)
        result = np.array(result).ravel()
        result[result == 0] = -1
        return result
