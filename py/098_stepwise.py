#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:45:44 2018

@author: Kazuki
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
#from glob import glob
import count
import utils, utils_cat
#utils.start(__file__)
#==============================================================================

SEED = 71

params = {
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




use_files = ['train_f001', 
#             'train_f002_WEEKDAY_APPR_PROCESS_START-ORGANIZATION_TYPE',
#             'train_f002_OCCUPATION_TYPE-ORGANIZATION_TYPE'
             ]


# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


CAT = list( set(X.columns) & set(utils_cat.ALL) )

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
model = lgb.train(params, dtrain, 1000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])

features_search = imp[imp['split']>0].feature.tolist()
features_curr = features_search[:20]

# =============================================================================
# stepwise
# =============================================================================

ex.stepwise(params, X, y, features_search, features_curr, best_score=0, 
            send_line=utils.send_line,
             eval_key='auc-mean', maximize=True,
             num_boost_round=9999, 
             folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, fobj=None, 
             feval=None, init_model=None, feature_name='auto', categorical_feature=CAT, 
             esr=None, fpreproc=None, verbose_eval=None, show_stdv=True, 
             seed=0, callbacks=None)






#==============================================================================
utils.end(__file__)




