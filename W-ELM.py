
# coding: utf-8

# In[41]:

import numpy as np
import itertools
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from twelm import TWELM, XELM, RBFNet, EEM



from sklearn.metrics import confusion_matrix



# In[29]:

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


# In[30]:

def process_cm(confusion_mat, i=0):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    return TP, FP, FN, TN


# In[31]:

def bac_error(Y, Y_predict):
    cm = confusion_matrix(Y, Y_predict)
    bac_values = np.zeros(cm.shape[0])
    for i in xrange(cm.shape[0]):
        tp, fp, fn, tn = process_cm(cm, i=i)
        if tp+fn > 0 and tn+fp>0:
            bac_values[i] = 0.5*tp/(tp+fn) + 0.5*tn/(tn+fp)
    return bac_values


# In[32]:

def bac_scorer(estimator, X, Y):
    Y_predict = estimator.predict(X)
    bac_values = bac_error(Y, Y_predict)
    return np.mean(bac_values)


# In[33]:

def perform_grid_search(estimator, features, activity,scorer,param_grid, n_outer_folds, n_inner_folds, n_outer_repetitions):
    """
    returns
    test score for each outer fold
    best score from grid search
    best estimator parameters for each iteration
    """
    test_scores = np.zeros(n_outer_folds*n_outer_repetitions)
    train_scores = np.zeros(n_outer_folds*n_outer_repetitions)
    best_parameters = []
    for rep in range(n_outer_repetitions):
        #print('%d/%d' % (rep, n_outer_repetitions))
        fold = StratifiedKFold(activity, n_folds=n_outer_folds, shuffle=True)
        #print(len(fold))
        fit_params = {}
        if isinstance(estimator, XELM) and 'h' in param_grid:
            max_h = min(features.shape[0], max(param_grid['h']))
            fit_params['hidden_layer'] = features[np.random.choice(features.shape[0],
                                            max_h,
                                            replace=False)]
        for i, (train_index, test_index) in enumerate(fold):
            search = GridSearchCV(estimator,                                  param_grid,                                  scoring=scorer,
                                  cv=n_inner_folds,
                                  n_jobs=1,
                                  fit_params=fit_params)

            search.fit(features[train_index], activity[train_index])

            test_score = scorer(search.best_estimator_, features[test_index], activity[test_index])
            test_scores[rep*n_outer_folds+i] = test_score
            train_scores[rep*n_outer_folds+i] = search.best_score_

            best_parameters.append(search.best_params_)
#             print('train score=%f, test score=%f' % (search.best_score_, test_score))
            print(search.best_params_)
    return test_scores, train_scores, best_parameters


# In[34]:

def test_models(estimators, estimator_grids, X, Y, scorer, n_outer_folds, n_inner_folds, n_outer_repetitions):
    estimator_scores = np.zeros(len(estimators))
    estimator_scores_std = np.zeros(len(estimators))
    assert len(estimators) == len(estimator_grids)
    for i, (estimator, grid) in enumerate(itertools.izip(estimators, estimator_grids)):
        print(type(estimator).__name__)
        
        scores_test, _, _ = perform_grid_search(estimator,
                                                X,
                                                Y,
                                                param_grid=grid,
                                                scorer=scorer,
                                                n_outer_folds=n_outer_folds,
                                                n_inner_folds=n_inner_folds, 
                                                n_outer_repetitions=n_outer_repetitions)
        print(scores_test)
        estimator_scores[i]= scores_test.mean()
        estimator_scores_std[i] = scores_test.std()
    
    return estimator_scores, estimator_scores_std
        


# In[44]:

estimators = [
    EEM(h=100, f='tanimoto'),
	EEM(h=100, f='kulczynski2'),
    RBFNet(h=100),
    XELM(h=10, f='tanimoto', balanced=True),
    XELM(h=10, f='kulczynski2', balanced=True),
    RandomForestClassifier(n_jobs=-1)
]

estimator_grids = [
    {'C': [ None, 100000], 'h': [ 500, 1500]},
	{'C': [ None, 100000], 'h': [ 500, 1500]},
    {'C': [ 1000, 100000], 'h': [ 500,  1500], 'b':[0.4, 1, 2]},
    {'C': [ 1000, 100000], 'h': [ 500,  1500]},
	{'C': [ 1000, 100000], 'h': [ 500,  1500]},
    {'n_estimators': [75, 125]}
]

estimator_grids_simple = [
    {'C': [ None], 'h': [ 500, 1500]},
	{'C': [ None], 'h': [ 500, 1500]},
    {'C': [ 1000], 'h': [   1500], 'b':[0.4,  2]},
    {'C': [ 1000], 'h': [   1500]},
	{'C': [ 1000], 'h': [   1500]},
    {'n_estimators': [75, 125]}
]



# In[46]:

def get_estimator_descritpion(estimator):
    name = type(estimator).__name__
    if hasattr(estimator, 'metric_name') and not isinstance(estimator, RBFNet):
        name+= '(%s)' % estimator.metric_name
    return name

scores_grid = pd.DataFrame()
scores_grid.loc[:, 'dataset'] = pd.Series()
for estimator in estimators:
    scores_grid.loc[:,get_estimator_descritpion(estimator) +'_score'] = pd.Series(np.zeros(len(datafiles)))
    scores_grid.loc[:, get_estimator_descritpion(estimator)+'_std'] = pd.Series(np.zeros(len(datafiles)))

for i, datafile in enumerate(datafiles):
    print(datafile)
    
    scores_grid.ix[i, 'dataset'] = datafile
    features, activity = load_svmlight_file(datafile)

    
    estimators_scores, estimatoirs_scores_std = test_models(estimators, 
                                                            estimator_grids,
                                                            features,
                                                            activity,
                                                            scorer=bac_scorer,
                                                            n_outer_folds=2,
                                                            n_inner_folds=3,
                                                            n_outer_repetitions=2)
    
    
    for estimator, score, std in itertools.izip(estimators, estimators_scores, estimatoirs_scores_std):
        scores_grid.ix[i, get_estimator_descritpion(estimator)+'_score'] = score
        scores_grid.ix[i, get_estimator_descritpion(estimator)+'_std'] = std
    
    


# In[13]:

#scores_grid


# In[14]:

scores_grid.to_csv(r'data/scores_grid.csv', index=False)


# a)
#     1. outer*n
#     2. fix projection, adjust size
#     result: std minimization
# b) Models then projections then representations
# 
# c) github

# In[ ]:



