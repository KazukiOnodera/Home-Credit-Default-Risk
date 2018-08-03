#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 04:36:45 2018

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
from collections import defaultdict
import utils
utils.start(__file__)
#==============================================================================

LOOP = 3

SEED = 71

NROUND = 4800


SUBMIT_FILE_PATH = '../output/804-1.csv.gz'

COMMENT = f'CV auc-mean(7-fold): 0.80520 + 0.00384 all(1249)'

EXE_SUBMIT = True

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

np.random.seed(SEED)

Categorical = ['FLAG_DOCUMENT_PATTERN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
               'HOUSETYPE_MODE', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
               'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE',
               'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START']

# =============================================================================
# load data
# =============================================================================
X_train = pd.read_pickle('../feature_someone/0803_LB----_CV0805/20180803_train_rk.pkl').drop('TARGET', axis=1)
y = utils.read_pickles('../data/label').TARGET

col_var = [c for c in X_train.columns if c.endswith('_var')]
col_drop = [c for c in col_var if c.replace('_var', '_std') in X_train.columns]
X_train.drop(col_drop, axis=1, inplace=True)


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[ X_train.columns.duplicated() ] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()

CAT = list( set(X_train.columns)&set(Categorical) )

print(f'category: {CAT}')

X_test = pd.read_pickle('../feature_someone/0803_LB----_CV0805/20180803_test_rk.pkl').drop('TARGET', axis=1)
X_test.drop(col_drop, axis=1, inplace=True)
X_test = X_test[X_train.columns]

X_train.head().to_csv(SUBMIT_FILE_PATH.replace('.csv', '_X.csv'),
       index=False, compression='gzip')

# =============================================================================
# training
# =============================================================================
dtrain = lgb.Dataset(X_train, y, categorical_feature=CAT )

models = []
for i in range(LOOP):
    print(f'LOOP: {i}')
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND,
                      categorical_feature=CAT)
#    model.save_model(f'lgb{i}.model')
    models.append(model)

imp = ex.getImp(models)
imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# =============================================================================
# predict
# =============================================================================
sub = pd.read_pickle('../data/sub.p')

gc.collect()

label_name = 'TARGET'

sub[label_name] = 0
for model in models:
    y_pred = model.predict(X_test)
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




