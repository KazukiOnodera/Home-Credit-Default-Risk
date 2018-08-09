#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:15:58 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from glob import glob

import GP
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


import utils, utils_best
#utils.start(__file__)
#==============================================================================

SEED = 71


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.9,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }


loader = utils_best.Loader('LB804')

# =============================================================================
# load
# =============================================================================
X = loader.train()
y = utils.read_pickles('../data/label').TARGET

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(loader.category()) )

X = pd.get_dummies(X, columns=CAT)

# inf to max+1
for c in tqdm(X.columns):
    X[c].replace(np.inf, X[c].replace(np.inf, 0).max()+1, inplace=True)

# -inf to max+1
for c in tqdm(X.columns):
    X[c].replace(-np.inf, X[c].replace(-np.inf, 0).min()-1, inplace=True)

# na to min -1
for c in tqdm(X.columns):
    X[c].fillna(X[c].min()-1, inplace=True)

# =============================================================================
# 
# =============================================================================

#def get_folds():
#    folds_list = []
#    for i in range(2):
#        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
#        folds = skf.split(X=np.zeros(X.shape[0]), y=y)
#        folds_list.append(folds)
#    return folds_list
#
#clf = LogisticRegression()
#clf = RandomForestClassifier(
#                             n_estimators=500,
#                             max_features=0.8,
#                             max_depth=5,
#                             min_samples_leaf=150,
#                             n_jobs=-1,
#                             )


#def get_fitness(feature_new, clf=clf):
#    if isinstance(feature_new, int) or isinstance(feature_new, float):
#        return 0
#    if isinstance(feature_new, np.ndarray):
#        feature_new = pd.Series(feature_new)
#    folds_list = get_folds()
#    X_ = pd.concat([X, feature_new], axis=1)
#    y_pred_all = pd.Series(0, index=y.index)
#    for folds in folds_list:
#        y_pred = pd.Series(index=y.index)
#        for train_idx, test_idx in folds:
#            X_train = X_.iloc[train_idx]
#            y_train = y.iloc[train_idx]
#            X_valid = X_.iloc[test_idx]
#            clf.fit(X_train, y_train)
#            y_pred.iloc[test_idx] = clf.predict_proba(X_valid)[:,1]
#        y_pred_all += y_pred.rank()
#    y_pred_all /= y_pred_all.max()
#    f = roc_auc_score(y, y_pred_all)
#    print(f)
#    return f

def get_fitness_logloss(feature_new):
    if isinstance(feature_new, int) or isinstance(feature_new, float):
        return 999
    if isinstance(feature_new, np.ndarray):
        feature_new = pd.Series(feature_new)
    if not isinstance(feature_new, pd.Series):
        return 999 # np.float64
    if feature_new.isnull().sum()>0 or feature_new.var()==0:
        return 999
    f = log_loss(y, feature_new.map(np.tanh))
    print(f)
    return f

# =============================================================================
# optimize
# =============================================================================
gp = GP.GP(X, y, 2, -3, 3, population=100, generation=10,
        feval=get_fitness_logloss, maximize=False, n_jobs=10)

gp.fit()

print(gp[0].eval())
print(gp[0].fitness, gp[0].parse())




#==============================================================================
utils.end(__file__)


