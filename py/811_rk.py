#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 02:00:02 2018

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
from glob import glob
#import count
import utils, utils_cat
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


Categorical = ['FLAG_DOCUMENT_PATTERN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
               'HOUSETYPE_MODE', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
               'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE',
               'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START']

# =============================================================================
# load
# =============================================================================
X = pd.read_pickle('../feature_someone/0803_LB----_CV0805/20180803_train_rk.pkl').drop('TARGET', axis=1)
y = utils.read_pickles('../data/label').TARGET

col_var = [c for c in X.columns if c.endswith('_var')]
col_drop = [c for c in col_var if c.replace('_var', '_std') in X.columns]
X.drop(col_drop, axis=1, inplace=True)


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(Categorical) )

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)


#==============================================================================
utils.end(__file__)
#utils.stop_instance()




