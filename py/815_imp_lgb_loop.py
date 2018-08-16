#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:33:01 2018

@author: Kazuki
"""


import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
#argv = sys.argv
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
#from collections import defaultdict
#from glob import glob
#import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

#SEED = int(argv[1])


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



imp = pd.read_csv('LOG/imp_801_imp_lgb.py_s111_s72.csv')
imp.sort_values('total', ascending=False, inplace=True)

files = ('../feature/train_' + imp.head(3000).feature + '.f').tolist()

# =============================================================================
# all data
# =============================================================================

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

#X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')

col_drop = ['f750_y_pred',
'f510_SK_ID_BUREAU',
'f509_SK_ID_BUREAU',
'f001_AMT_CREDIT-d-AMT_ANNUITY',
'f001_PAYMENT_RATE',
'nejumi',
'f001_AMT_GOODS_PRICE-d-AMT_ANNUITY',
'f601_Closed_STATUS_1_var',
'f601_Closed_STATUS_X_var',
'f001_AMT_GOODS_PRICE-d-AMT_CREDIT',
'f105_prevapp_future_payment_19m',
'f001_AMT_GOODS_PRICE-m-AMT_CREDIT',
'f001_AMT_CREDIT-d-CNT_CHILDREN',
'f602_Active_CURR-BUREAU_cnt_var',
'f105_amt_unpaid_sum-p-app',
'f105_prevapp_future_payment_13m',
'f001_AMT_CREDIT-d-CNT_FAM_MEMBERS',
'f001_AMT_REQ_CREDIT_BUREAU_QRT',
'f001_AMT_ANNUITY-d-cnt_adults',
'f701_all_credit-prevact_min',
'f001_AMT_ANNUITY-d-CNT_CHILDREN',
'f001_EXT_SOURCES_std']

col_drop = list( set(col_drop) & set(X.columns) )

X.drop(col_drop, axis=1, inplace=True)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'CAT: {CAT}')

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT, free_raw_data=False)
gc.collect()

for i in range(10):
    gc.collect()
    seed = np.random.randint(99999)
    print('seed:', seed)
    param['seed'] = seed
    ret, models = lgb.cv(param, dtrain, 9999, nfold=6,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=seed)
    
    result = f"CV auc-mean({seed}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
    print(result)
    
    utils.send_line(result)
    imp = ex.getImp(models)
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    
    imp.sort_values('total', ascending=False, inplace=True)
    imp.reset_index(drop=True, inplace=True)
    
    
    imp.to_csv(f'LOG/imp_{__file__}-s{seed}.csv', index=False)


#==============================================================================
utils.end(__file__)
utils.stop_instance()

