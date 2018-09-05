#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:00:47 2018

@author: Kazuki
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from glob import glob
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)

NFOLD = 7

HEADS = list(range(500, 2300, 100))

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

# =============================================================================
# load
# =============================================================================
imp = pd.read_csv('LOG/imp_803_adv_imp_lgb.py.csv')

imp.sort_values('total', ascending=False, inplace=True)

y = utils.read_pickles('../data/label').TARGET



for HEAD in HEADS:
    files = ('../feature/train_' + imp.head(HEAD).feature + '.f').tolist()
    
#    files = utils.get_use_files(use_files, True)
    
    X = pd.concat([
                    pd.read_feather(f) for f in tqdm(files, mininterval=60)
                   ], axis=1)
    
    if X.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X.shape {X.shape}')
    
    gc.collect()
    
    CAT = list( set(X.columns)&set(utils_cat.ALL))
    
    # =============================================================================
    # cv
    # =============================================================================
    dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
    gc.collect()
    
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    result = f"CV auc-mean({SEED}:{HEAD}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
    print(result)
    
    utils.send_line(result)



#==============================================================================
utils.end(__file__)
utils.stop_instance()



