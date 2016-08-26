import cProfile
from io import BytesIO
import pstats
from conda.utils import memoized
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from theano import ifelse
from theano.sandbox.linalg import matrix_inverse
from twelm import XELM, EEM, RBFNet

from scipy.sparse import csr_matrix, issparse
import numpy as np
from theano.sandbox.linalg.ops import pinv, inv_as_solve
import theano.tensor as T
# import theano.sparse
import theano

__author__ = 'amyronov'


def tanimoto_theano(X, W):
    """ Tanimoto similarity function """
    XW = T.dot(X, W.T)
    XX = T.abs_(X).sum(axis=1).reshape((-1, 1))
    WW = T.abs_(W).sum(axis=1).reshape((1, -1))
    return XW / (XX + WW - XW)


def kulczynski2_theano(X, W, b=None):
    """ Kulczynski2 similarity function """
    XW = T.dot(X, W.T)
    XX = T.abs_(X).sum(axis=1).reshape((-1, 1))
    WW = T.abs_(W).sum(axis=1).reshape((1, -1))
    return 0.5 * (XW / XX + XW / WW)


def f1_score_theano(X, W, b=None):
    XW = T.dot(X, W.T)
    XX = T.abs_(X).sum(axis=1).reshape((-1, 1))
    WW = T.abs_(W).sum(axis=1).reshape((1, -1))
    return T.inv(XW / XX) + T.inv(XW / WW)


def kulczynski3_theano(X, W, b=None):
    """
    GMean of precision and recall
    """
    XW = T.dot(X, W.T)
    XX = T.abs_(X).sum(axis=1).reshape((-1, 1))
    WW = T.abs_(W).sum(axis=1).reshape((1, -1))
    return T.sqrt((XW / XX) * (XW / WW))


def euclidean_theano(X, W, b=None):
    squared_euclidean_distances = (X ** 2).sum(1).reshape((X.shape[0], 1)) + (W ** 2).sum(
        1).reshape((1, W.shape[0])) - 2 * X.dot(W.T)
    return T.sqrt(T.abs_(squared_euclidean_distances))


metric_theano = {
    'tanimoto': tanimoto_theano,
    'kulczynski2': kulczynski2_theano,
    'euclidean': euclidean_theano,
    'kulczynski3': kulczynski3_theano,
    'f1_score': f1_score_theano,

}


@memoized
def get_xelm_learning_function(f_name):
    # global xelm_learning_function


    X_matrix = T.dmatrix('X')
    W_matrix = T.dmatrix('W')
    # b_vector = T.dvector('b')
    w_vector = T.dvector('w')
    C_scalar = T.scalar('C')
    y_vector = T.dvector('y')

    H_matrix = metric_theano[f_name](X_matrix, W_matrix)

    Hw_matrix = H_matrix * w_vector.reshape((-1, 1))
    yw_vector = (y_vector * w_vector)

    beta_matrix = T.dot(
        matrix_inverse(T.dot(Hw_matrix.T, Hw_matrix) + 1.0 / C_scalar * T.eye(Hw_matrix.shape[1])),
        T.dot(Hw_matrix.T, yw_vector).T)
    # beta_function = theano.function([H_matrix, C_scalar, y_vector], beta_matrix)
    xelm_learning_function = theano.function([X_matrix, W_matrix, w_vector, C_scalar, y_vector],
                                             beta_matrix)
    return xelm_learning_function


@memoized
def get_xelm_predict_function(f_name):
    X_matrix = T.dmatrix('X')
    W_matrix = T.dmatrix('W')
    beta = T.dmatrix('beta')

    H_matrix = metric_theano[f_name](X_matrix, W_matrix)
    s = T.sgn(T.dot(H_matrix, beta))

    xelm_predict_function = theano.function([X_matrix, W_matrix, beta], s)
    return xelm_predict_function


class XELMTheano(XELM):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # if kwargs['f'] is str:
        #     self.f = metric_theano['f']
        self.predict_function = get_xelm_predict_function(kwargs['f'])
        self.learning_function = get_xelm_learning_function(self.metric_name)

    # def fit2(self, X, y, hidden_layer=None):
    #     self.W, self.b = self._hidden_init(X, y, hidden_layer)
    #     H = self.f(X, self.W, self.b)
    #
    #     if self.balanced:
    #         counts = {l: float(y.tolist().count(l)) for l in set(y)}
    #         ms = max([counts[k] for k in counts])
    #         self.counts = {l: np.sqrt(ms / counts[l]) for l in counts}
    #     else:
    #         self.counts = {l: 1 for l in set(y)}
    #
    #     # print(H.shape)
    #
    #     w = np.array([[self.counts[a] for a in y]]).T
    #     H = np.multiply(H, w)
    #     y = np.multiply(y.reshape(-1, 1), w).ravel()
    #
    #     # print(H.shape, w.shape, y.shape)
    #
    #     self.beta = beta_function(H, self.C, y).reshape(-1, 1)
    #     # print(self.beta)

    def fit(self, X, y, hidden_layer=None):

        # if self.learning_function is None:
        #     self.learning_function = theano_init(self.metric_name)

        self.W, self.b = self._hidden_init(X, y, hidden_layer)
        if issparse(self.W):
            self.W = self.W.toarray()

        if issparse(X):
            X = X.toarray()

        if self.balanced:
            counts = {l: float(y.tolist().count(l)) for l in set(y)}
            ms = max([counts[k] for k in counts])
            self.counts = {l: np.sqrt(ms / counts[l]) for l in counts}
        else:
            self.counts = {l: 1 for l in set(y)}
        w = np.array([[self.counts[a] for a in y]]).ravel()

        # print(w.shape, y.shape, self.C)

        self.beta = self.learning_function(X, self.W, w.T, self.C, y).reshape(
            -1, 1)

    def predict(self, X):
        if issparse(X):
            X = X.toarray()
        return self.predict_function(X, self.W, self.beta)


@memoized
def get_eem_learning_function(metric_name):
    W = T.dmatrix('W')
    X = T.dmatrix('X')
    H = metric_theano[metric_name](X, W)
    H_func = theano.function([X, W], H)
    C = T.scalar('C')

    H_plus = T.dmatrix('H_plus')
    H_minus = T.dmatrix('H_minus')

    sigma_plus = T.dmatrix('sigma_plus')
    sigma_minus = T.dmatrix('sigma_minus')

    sigma_plus_reg = sigma_plus + T.eye(sigma_plus.shape[1]) / 2 * C
    sigma_minus_reg = sigma_minus + T.eye(sigma_minus.shape[1]) / 2 * C

    m_plus = H_plus.mean(axis=0).T
    m_minus = H_minus.mean(axis=0).T

    mean_diff = m_plus - m_minus
    beta = (2 * T.dot(matrix_inverse(sigma_plus_reg + sigma_minus_reg), mean_diff)
            / mean_diff.norm(L=2))
    func = theano.function([H_plus, H_minus, sigma_plus, sigma_minus, C],
                           [beta, sigma_plus_reg, sigma_minus_reg, m_plus, m_minus])

    def eem_learning_function(X, W, y, C):
        the_H = H_func(X, W)

        the_H_plus = the_H[y == 1]
        the_H_minus = the_H[y == -1]

        the_sigma_plus = LedoitWolf(store_precision=False).fit(the_H_plus).covariance_
        the_sigma_minus = LedoitWolf(store_precision=False).fit(the_H_minus).covariance_

        if C is None:
            C = 0

        return func(the_H_plus, the_H_minus, the_sigma_plus, the_sigma_minus, C)

    return eem_learning_function


@memoized
def get_eem_predict_function(metric_name):
    W = T.dmatrix('W')
    X = T.dmatrix('X')

    beta = T.dmatrix('beta')
    m_plus = T.dmatrix('m_plus')
    m_minus = T.dmatrix('m_minus')
    sigma_plus = T.dmatrix('sigma_plus')
    sigma_minus = T.dmatrix('sigma_minus')

    H = metric_theano[metric_name](X, W)

    def gaussian(x, mu, sigma):
        return T.exp(T.power((x - mu[0]), 2) / (-2 * sigma)[0]) / (sigma * T.sqrt(2 * np.pi))[0]

    x = T.dot(H, beta)
    r_plus = gaussian(x, T.dot(beta.T, m_plus),
                      T.dot(T.dot(beta.T, sigma_plus), beta))
    r_minus = gaussian(x, T.dot(beta.T, m_minus),
                       T.dot(T.dot(beta.T, sigma_minus), beta))

    result = T.argmax(T.horizontal_stack(r_minus, r_plus), axis=1)

    eem_predict_function = theano.function(
        [X, W, beta, m_plus, m_minus, sigma_plus, sigma_minus], result)
    return eem_predict_function


class EEMTheano(EEM):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.learning_function = get_eem_learning_function(self.metric_name)
        self.predict_function = get_eem_predict_function(self.metric_name)
        self.b = None

    def fit(self, X, y, hidden_layer=None):
        self.W, _ = self._hidden_init(X, y, hidden_layer)
        if issparse(self.W):
            self.W = self.W.toarray()
        if issparse(X):
            X = X.toarray()
        self.beta, self.sigma_plus, self.sigma_minus, \
        self.m_plus, self.m_minus = self.learning_function(X, self.W, y, self.C)

        self.beta = self.beta.reshape(-1, 1)
        self.m_plus = self.m_plus.reshape(-1, 1)
        self.m_minus = self.m_minus.reshape(-1, 1)
        pass

    def predict(self, X):
        if issparse(X):
            X = X.toarray()

        result = self.predict_function(X, self.W, self.beta, self.m_plus, self.m_minus,
                                       self.sigma_plus, self.sigma_minus)
        result = result.ravel()
        result[result == 0] = -1
        return result


def get_rbfnet_learning_func(f_name):
    assert f_name == 'euclidean'
    X_matrix = T.dmatrix('X')
    W_matrix = T.dmatrix('W')
    b = T.scalar('b')
    C_scalar = T.scalar('C')
    y_vector = T.dvector('y')

    H_matrix = metric_theano[f_name](X_matrix, W_matrix)
    H_rbf = np.exp(T.power(H_matrix, 2) * (-b))

    beta_matrix = T.dot(
        matrix_inverse(T.dot(H_rbf.T, H_rbf) + 1.0 / C_scalar * T.eye(H_rbf.shape[1])),
        T.dot(H_rbf.T, y_vector).T)
    # beta_function = theano.function([H_matrix, C_scalar, y_vector], beta_matrix)
    rbfnet_learning_function = theano.function([X_matrix, W_matrix, C_scalar, b, y_vector],
                                               beta_matrix)
    return rbfnet_learning_function


@memoized
def get_rbfnet_predict_function(metric_name):
    X_matrix = T.dmatrix('X')
    W_matrix = T.dmatrix('W')
    beta = T.dvector('beta')
    b = T.scalar('b')

    H_matrix = metric_theano[metric_name](X_matrix, W_matrix)
    H_rbf = np.exp(T.power(H_matrix, 2) * (-b))
    s = T.sgn(T.dot(H_rbf, beta))

    rbfnet_predict_function = theano.function([X_matrix, W_matrix, beta, b], s)
    return rbfnet_predict_function


class RBFNetTheano(RBFNet):
    def __init__(self, *args, **kwargs):
        super(RBFNetTheano, self).__init__(*args, **kwargs)

        self.learning_function = get_rbfnet_learning_func(self.metric_name)
        self.predict_function = get_rbfnet_predict_function(self.metric_name)

    def fit(self, X, y, hidden_layer=None):
        self.W, _ = self._hidden_init(X, y, hidden_layer)
        if issparse(self.W):
            self.W = self.W.toarray()

        if issparse(X):
            X = X.toarray()

        self.beta = self.learning_function(X, self.W, self.C, self.b, y)

    def predict(self, X):
        if issparse(X):
            X = X.toarray()
        result = self.predict_function(X, self.W, self.beta, self.b)
        return result


if __name__ == '__main__':
    datafile = r'data/hiv_integrase_ExtFP.libsvm'
    # datafile = r'data/d2_ExtFP.libsvm'
    # datafile = r'data/diabetes_scale.libsvm'
    features, activity = load_svmlight_file(datafile)

    print('data loaded')


    def compare_2_models(model1, model2, X, y, h):
        h = min(X.shape[0], h)
        hidden_layer = features[np.random.choice(X.shape[0],
                                                 h,
                                                 replace=False)]
        print('training 1st model')
        pr = cProfile.Profile()
        pr.enable()
        model1.fit(X, y, hidden_layer=hidden_layer)
        y1 = model1.predict(X)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats()

        print('training 2nd model')
        pr = cProfile.Profile()
        pr.enable()
        model2.fit(X, y, hidden_layer=hidden_layer)
        y2 = model2.predict(X)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats()

        print(f1_score(y, y2))
        print(f1_score(y, y1))

        return np.allclose(y1, y2)


    h = 1000

    print(compare_2_models(

        EEM(h=h, f='tanimoto', C=1000, random_state=0),
        EEMTheano(h=h, f='tanimoto', C=1000, random_state=0),
        features,
        activity,
        h))
