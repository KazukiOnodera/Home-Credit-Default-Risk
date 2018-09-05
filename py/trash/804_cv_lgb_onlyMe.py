#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 01:29:16 2018

@author: Kazuki
"""

import gc, os
from tqdm import tqdm
import pandas as pd
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

SEEDs = list(range(75, 78))

HEADS = list(range(300, 1000, 100))

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
#         'seed': SEED
         }

# =============================================================================
# load
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb_onlyMe.py-2.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)

files = ('../feature/train_' + imp.head(max(HEADS)).feature + '.f').tolist()

#files = utils.get_use_files(use_files, True)

X_all = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET




for SEED in SEEDs:
    param.update({'seed':SEED})
    
    for HEAD in HEADS:
        
        X = X_all.iloc[:, :HEAD]
        
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
        
        ret = lgb.cv(param, dtrain, 9999, nfold=5,
                     early_stopping_rounds=100, verbose_eval=None,
                     seed=SEED)
        
        result = f"CV auc-mean({SEED}:{HEAD}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
        print(result)
        
        utils.send_line(result)



#==============================================================================
utils.end(__file__)
#utils.stop_instance()


