#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:40:45 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
from glob import glob
import lightgbm as lgb
import multiprocessing
import gc
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 5

NROUND = 1700

SUBMIT_FILE_PATH = '../output/606-3.csv.gz'

COMMENT = 'add 302'

EXE_SUBMIT = True

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 255,
         'max_bin': 255,
         'colsample_bytree': 0.5,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
#         'verbose':-1,
         }


categorical_feature = ['NAME_CONTRACT_TYPE',
#                     'CODE_GENDER',
#                     'FLAG_OWN_CAR',
#                     'FLAG_OWN_REALTY',
                     'NAME_TYPE_SUITE',
                     'NAME_INCOME_TYPE',
                     'NAME_EDUCATION_TYPE',
                     'NAME_FAMILY_STATUS',
                     'NAME_HOUSING_TYPE',
                     'OCCUPATION_TYPE',
                     'WEEKDAY_APPR_PROCESS_START',
                     'ORGANIZATION_TYPE',
                     'FONDKAPREMONT_MODE',
                     'HOUSETYPE_MODE',
                     'WALLSMATERIAL_MODE',
#                     'EMERGENCYSTATE_MODE'
                     ]

print(f'seed: {SEED}')
np.random.seed(SEED)


# =============================================================================
# train
# =============================================================================
folders = sorted(glob('../data/*_train'))
X_train = pd.concat([
                utils.read_pickles(f) for f in (folders)
                ], axis=1)
y_train = utils.read_pickles('../data/label').TARGET

print(f'categorical_feature: {categorical_feature}')
print(f'folders: {folders}')

dtrain = lgb.Dataset(X_train, y_train, 
                     categorical_feature=categorical_feature)

del X_train, y_train; gc.collect()



models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND,
                      categorical_feature=categorical_feature)
    model.save_model(f'lgb{i}.model')
    models.append(model)
    
del dtrain; gc.collect()

"""

models = []
for i in range(3):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""


# =============================================================================
# test
# =============================================================================
folders = sorted(glob('../data/*_test'))

dtest = pd.concat([
                utils.read_pickles(f) for f in (folders)
                ], axis=1)

sub = pd.read_pickle('../data/sub.p')

gc.collect()

label_name = 'TARGET'

sub[label_name] = 0
for model in models:
    y_pred = model.predict(dtest)
    sub[label_name] += pd.Series(y_pred).rank()
sub[label_name] /= LOOP
sub[label_name] /= sub[label_name].max()
sub['SK_ID_CURR'] = sub['SK_ID_CURR'].map(int)

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)


#==============================================================================
utils.end(__file__)


