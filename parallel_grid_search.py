from collections import Sized
from functools import partial
from time import sleep
import itertools
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable

import ipyparallel as ipp
from loader import loader
import numpy as np

__author__ = 'amyronov'

from sklearn.grid_search import GridSearchCV, _CVScoreTuple, ParameterGrid


def my_fit_and_score(train_test_parameters,
                     estimator=None,
                     X=None,
                     y=None,
                     verbose=False,
                     fit_params=None,
                     return_parameters=True,
                     scorer=None,
                     x_is_index=True,
                     names=('X', 'y')):
    from runner import bac_scorer, bac_error, confusion_matrix, process_cm

    train, test, parameters = train_test_parameters

    if x_is_index:
        index = X
        X = None
    if X is None:
        if 'X' in globals():
            X = globals()[names[0]]
            y = globals()[names[1]]
        else:
            X, y = loader(names[0], names[1])()
            globals()[names[0]] = X
            globals()[names[1]] = y

    if x_is_index:
        X = X[index]
        y = y[index]

    return _fit_and_score(estimator=estimator,
                          X=X,
                          y=y,
                          verbose=verbose,
                          parameters=parameters,
                          fit_params=fit_params,
                          return_parameters=return_parameters,
                          train=train,
                          test=test,
                          scorer=bac_scorer)


def clear_globals(names):
    for name in names:
        if name in globals():
            globals()['name'] = None


class GridSearchCVParallel(GridSearchCV):
    def __init__(self, *args, **kwargs):
        if 'view' in kwargs:
            self.view = kwargs['view']
            del kwargs['view']
        if self.view is None:
            self.view = ipp.Client().load_balanced_view()

        if 'callback' in kwargs:
            self.callback = kwargs['callback']
            del kwargs['callback']
        else:
            self.callback = None

        super(GridSearchCVParallel, self).__init__(*args, **kwargs)

    def fit(self, X, y=None, x_is_index=False, X_name='X', y_name='y'):

        parameter_iterable = ParameterGrid(self.param_grid)
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)

        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)


        # out = Parallel(
        #     n_jobs=self.n_jobs, verbose=self.verbose,
        #     pre_dispatch=pre_dispatch
        # )(
        #     delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
        #                             train, test, self.verbose, parameters,
        #                             self.fit_params, return_parameters=True,
        #                             error_score=self.error_score)
        #         for parameters in parameter_iterable
        #         for train, test in cv)

        train_test_parameters = ((train, test, parameters) \
                                 for parameters in parameter_iterable for train, test in cv)

        length = len(parameter_iterable) * len(cv)

        if x_is_index:
            X_to_pass = X
            y_to_pass = None
        else:
            X_to_pass = None
            y_to_pass = None

        self.view.block = False
        # print('sequences')

        # sequences = [
        #     train_test_parameters,
        #     [clone(base_estimator)] * length,
        #     [X_to_pass] * length,
        #     [y_to_pass] * length,
        #     [self.verbose] * length,
        #     [self.fit_params] * length,
        #     [True] * length,
        #     [self.scorer_] * length,
        #     [x_is_index] * length,
        # ]

        f = partial(my_fit_and_score, estimator=clone(base_estimator),
                    X=X_to_pass,
                    y=y_to_pass,
                    verbose=self.verbose,
                    fit_params=self.fit_params,
                    return_parameters=True,
                    scorer=None,
                    x_is_index=x_is_index,
                    names=(X_name, y_name))

        # print('before map')

        # import cProfile
        #
        # pr = cProfile.Profile()
        # pr.enable()
        chunksize = 10

        out = self.view.map(f, itertools.islice(train_test_parameters, 0, length),
                            ordered=False,
                            block=False,
                            chunksize=chunksize)  # length / len(self.view))
        # pr.disable()
        # pr.print_stats('cumulative')
        print('map called')
        if self.callback is not None:
            old_progress = out.progress
            while not out.ready():
                self.callback(out.progress * chunksize, length, out.elapsed)
                if old_progress == out.progress and out.progress > 0:
                    for id, info in self.view.queue_status(verbose=True).iteritems():
                        # print(id, info)
                        if isinstance(info, dict) and 'queue' in info and len(info['queue']) > 0:
                            print(id, info['queue'])

                    pass
                old_progress = out.progress
                sleep(10)
        print('map ready')
        out = out.get()


        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self
