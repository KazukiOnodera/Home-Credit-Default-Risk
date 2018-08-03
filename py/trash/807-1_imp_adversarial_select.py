#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:25:32 2018

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
from collections import defaultdict
#from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 10

FEATURE_SIZE = 5000

SAMPLE_SIZE = 80000


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
# all data
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)


files = ('../feature/train_' + imp.head(FEATURE_SIZE).feature + '.f').tolist()

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'CAT: {CAT}')

# =============================================================================
# def
# =============================================================================
WEIGHT = pd.read_feather('../feature/train_f750_y_pred.f').f750_y_pred
WEIGHT /= WEIGHT.sum()

def get_sample():
    ind = np.random.choice(X.index, p=WEIGHT, replace=False, size=SAMPLE_SIZE)
    
    return X.iloc[ind], y.iloc[ind]


# =============================================================================
# imp
# =============================================================================
models = []
for i in range(LOOP):
    X_, y_ = get_sample()
    dtrain = lgb.Dataset(X_, y_, categorical_feature=CAT )
    gc.collect()
    ret, models_ = lgb.cv(param, dtrain, 9999, nfold=7,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    models += models_

imp = ex.getImp(models).sort_values(['gain', 'feature'], ascending=[False, True])
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

"""
imp[imp.feature.str.startswith('f312_')]

__file__ = '801_imp_lgb.py'
"""
imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)


#==============================================================================
utils.end(__file__)
#utils.stop_instance()


