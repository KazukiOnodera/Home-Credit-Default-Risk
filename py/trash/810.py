#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:18:08 2018

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
#import count
import utils, utils_best
#utils.start(__file__)
#==============================================================================

SEED = 71

new_feature = 'f110'

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
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


loader = utils_best.Loader('CV805_LB803')

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

# =============================================================================
# stepwise
# =============================================================================

def save_df(df):
    df.to_csv('../data/tmp.csv', index=False)
    return


ex.stepwise(param, X, y, X.columns[:-1][::-1].tolist(), X.columns.tolist(), 
            best_score=0, send_line=None,
             eval_key='auc-mean', maximize=True, save_df=save_df, cv_loop=2,
             num_boost_round=100, 
             folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, fobj=None, 
             feval=None, init_model=None, feature_name='auto', categorical_feature=CAT, 
             esr=None, fpreproc=None, verbose_eval=None, show_stdv=True, 
             seed=2, callbacks=None)




#==============================================================================
utils.end(__file__)


