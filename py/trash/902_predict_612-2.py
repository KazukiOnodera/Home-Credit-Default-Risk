#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:46:37 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
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

NROUND = 2000

SUBMIT_FILE_PATH = '../output/612-2.csv.gz'

COMMENT = 'r2000'

EXE_SUBMIT = True

imp_file = 'LOG/imp_901_cv_612-2.py.csv'

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

categorical_feature = ['app_001_NAME_CONTRACT_TYPE',
                     'app_001_CODE_GENDER',
                     'app_001_FLAG_OWN_CAR',
                     'app_001_FLAG_OWN_REALTY',
                     'app_001_NAME_TYPE_SUITE',
                     'app_001_NAME_INCOME_TYPE',
                     'app_001_NAME_EDUCATION_TYPE',
                     'app_001_NAME_FAMILY_STATUS',
                     'app_001_NAME_HOUSING_TYPE',
                     'app_001_OCCUPATION_TYPE',
                     'app_001_WEEKDAY_APPR_PROCESS_START',
                     'app_001_ORGANIZATION_TYPE',
                     'app_001_FONDKAPREMONT_MODE',
                     'app_001_HOUSETYPE_MODE',
                     'app_001_WALLSMATERIAL_MODE',
                     'app_001_EMERGENCYSTATE_MODE']

# =============================================================================
# train
# =============================================================================
utils.check_feature()


print(f'seed: {SEED}')
np.random.seed(SEED)

if imp_file is None:
    remove_names = []
else:
    imp = pd.read_csv(imp_file).set_index('index')
    remove_names = imp[imp['split']==0].index.tolist()

files = sorted(glob('../feature/train*.f'))
remove_files = []
if len(remove_names)>0:
    for i in files:
        for j in remove_names:
            if i.endswith(j+'.f'):
                remove_files.append(i)
                break
    
    print(f'remove {len(remove_files)} files')
    files = sorted(list( set(files)-set(remove_files) ))
    print(f'read {len(files)} files')


X_train = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=100)
                ], axis=1)
y_train = utils.read_pickles('../data/label').TARGET

if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

dtrain = lgb.Dataset(X_train, y_train, 
                     categorical_feature=categorical_feature)
COL = X_train.columns.tolist()
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
for i in range(LOOP):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""


# =============================================================================
# test
# =============================================================================
files = sorted(glob('../feature/test*.f'))
remove_files = []
if len(remove_names)>0:
    for i in files:
        for j in remove_names:
            if i.endswith(j+'.f'):
                remove_files.append(i)
                break
    
    print(f'remove {len(remove_files)} files')
    files = sorted(list( set(files)-set(remove_files) ))
    print(f'read {len(files)} files')

dtest = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=100)
                ], axis=1)[COL]

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

