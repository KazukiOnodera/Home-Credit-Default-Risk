#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:01:50 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/home/kazuki_onodera/PythonLibrary')
#import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from glob import glob
import gc
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 10

NROUND = 1700

SUBMIT_FILE_PATH = '../output/711-1.csv.gz'

COMMENT = f'CV auc-mean: 0.80178 round: {NROUND}'

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

categorical_feature = ['f001_NAME_CONTRACT_TYPE',
                     'f001_CODE_GENDER',
                     'f001_FLAG_OWN_CAR',
                     'f001_FLAG_OWN_REALTY',
                     'f001_NAME_TYPE_SUITE',
                     'f001_NAME_INCOME_TYPE',
                     'f001_NAME_EDUCATION_TYPE',
                     'f001_NAME_FAMILY_STATUS',
                     'f001_NAME_HOUSING_TYPE',
                     'f001_OCCUPATION_TYPE',
                     'f001_WEEKDAY_APPR_PROCESS_START',
                     'f001_ORGANIZATION_TYPE',
                     'f001_FONDKAPREMONT_MODE',
                     'f001_HOUSETYPE_MODE',
                     'f001_WALLSMATERIAL_MODE',
                     'f001_EMERGENCYSTATE_MODE']


categorical_feature += ['f108_NAME_CONTRACT_TYPE',
                     'f108_WEEKDAY_APPR_PROCESS_START',
                     'f108_NAME_CASH_LOAN_PURPOSE',
                     'f108_NAME_CONTRACT_STATUS',
                     'f108_NAME_PAYMENT_TYPE',
                     'f108_CODE_REJECT_REASON',
                     'f108_NAME_TYPE_SUITE',
                     'f108_NAME_CLIENT_TYPE',
                     'f108_NAME_GOODS_CATEGORY',
                     'f108_NAME_PORTFOLIO',
                     'f108_NAME_PRODUCT_TYPE',
                     'f108_CHANNEL_TYPE',
                     'f108_NAME_SELLER_INDUSTRY',
                     'f108_NAME_YIELD_GROUP',
                     'f108_PRODUCT_COMBINATION']

categorical_feature += ['f109_NAME_CONTRACT_TYPE',
                         'f109_WEEKDAY_APPR_PROCESS_START',
                         'f109_NAME_CASH_LOAN_PURPOSE',
                         'f109_NAME_CONTRACT_STATUS',
                         'f109_NAME_PAYMENT_TYPE',
                         'f109_CODE_REJECT_REASON',
                         'f109_NAME_TYPE_SUITE',
                         'f109_NAME_CLIENT_TYPE',
                         'f109_NAME_GOODS_CATEGORY',
                         'f109_NAME_PORTFOLIO',
                         'f109_NAME_PRODUCT_TYPE',
                         'f109_CHANNEL_TYPE',
                         'f109_NAME_SELLER_INDUSTRY',
                         'f109_NAME_YIELD_GROUP',
                         'f109_PRODUCT_COMBINATION']

categorical_feature += ['FLAG_PHONE_PATTERN', 'FLAG_DOC_PATTERN'] # Maxwell

use_files = []
# =============================================================================
# train
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

# maxwell
maxwell = pd.read_feather('../feature_someone/Maxwell_train.f')
X = pd.concat([X, maxwell], axis=1); gc.collect()


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')
gc.collect()

CAT = list( set(X.columns)&set(categorical_feature))
print(f'category: {CAT}')

dtrain = lgb.Dataset(X, y, 
                     categorical_feature=CAT)
COL = X.columns.tolist()
del X, y, maxwell; gc.collect()



models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND,
                      categorical_feature=CAT)
#    model.save_model(f'lgb{i}.model')
    models.append(model)


del dtrain; gc.collect()

"""

models = []
for i in range(LOOP):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""


# =============================================================================
# test
# =============================================================================
files = utils.get_use_files(use_files, False)

dtest = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=100)
                ], axis=1)

# maxwell
maxwell = pd.read_feather('../feature_someone/Maxwell_test.f')
dtest = pd.concat([dtest, maxwell], axis=1)[COL]; gc.collect()


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


