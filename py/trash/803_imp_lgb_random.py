#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:56:51 2018

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
from multiprocessing import cpu_count, Pool
#from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
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

use_files = ['train_f'] # only me


# =============================================================================
# all data
# =============================================================================
files = utils.get_use_files(use_files, True)

X_all = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y_all = utils.read_pickles('../data/label').TARGET


if X_all.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_all.columns[X_all.columns.duplicated()] }')
print('no dup :) ')
print(f'X_all.shape {X_all.shape}')

gc.collect()

CAT = list( set(X_all.columns)&set(utils_cat.ALL))
print(f'CAT: {CAT}')

# =============================================================================
# imp
# =============================================================================
np.random.seed(SEED)

seeds = np.random.randint(9999, size=10)
for i,seed in enumerate(seeds):
    gc.collect()
    dtrain = lgb.Dataset(X_all.sample(frac=.3, random_state=seed), 
                         y_all.sample(frac=.3, random_state=seed), 
                         categorical_feature=CAT )
    model = lgb.train(param, dtrain, 2500)
    imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])
    
    """
    __file__ = '801_imp_lgb_onlyMe.py'
    """
    imp.to_csv(f'LOG/imp_{__file__}-{seed}.csv', index=False)


#==============================================================================
utils.end(__file__)


