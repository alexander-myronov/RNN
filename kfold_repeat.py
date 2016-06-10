import sklearn
from sklearn.cross_validation import StratifiedKFold

__author__ = 'amyronov'


class KFoldRepeat(object):
    def __init__(self, y, n_folds, n_reps):
        # super(KFoldRepeat, self).__init__(y, n_folds=n_folds)
        self.n_folds = n_folds
        self.n_reps = n_reps
        self.y = y
        self.rep = 0

    def __len__(self):
        return self.n_folds * self.n_reps

    def spawn_fold(self):
        return StratifiedKFold(self.y, n_folds=self.n_folds, shuffle=True)

    def spawn(self):
        for rep in xrange(self.n_reps):
            if rep % 10 == 0:
                pass  # print('kfold %d repetition' % rep)
            for tr, ts in self.spawn_fold():
                yield tr, ts

    def __iter__(self):
        # if self.rep == self.n_reps:
        #     raise StopIteration()
        return self.spawn()


def setup_kfold_patch(n_reps):
    kfold_mock = lambda y, n_folds: KFoldRepeat(y, n_folds, n_reps=n_reps)

    original = sklearn.cross_validation.StratifiedKFold
    sklearn.cross_validation.StratifiedKFold = kfold_mock

    def spawn_fold(self):
        # print('new fold')
        sklearn.cross_validation.StratifiedKFold = original
        kfold = StratifiedKFold(self.y, self.n_folds, shuffle=True)
        sklearn.cross_validation.StratifiedKFold = kfold_mock
        return kfold

    KFoldRepeat.spawn_fold = spawn_fold
