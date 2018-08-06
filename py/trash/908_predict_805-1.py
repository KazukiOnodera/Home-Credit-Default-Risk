#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:28:51 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
from collections import defaultdict
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
#import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from glob import glob
import utils, utils_cat, utils_best
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 3

NUM_CHANGE_DATA = 30

NROUND = 4500
NROUND = int(NROUND/NUM_CHANGE_DATA)

SUBMIT_FILE_PATH = '../output/805-1.csv.gz'

COMMENT = f'LB804 psuedo'

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
#         'seed': SEED
         }


np.random.seed(SEED)

loader = utils_best.Loader('LB804')

# =============================================================================
# load
# =============================================================================

X_train = loader.train()
y_train = utils.read_pickles('../data/label').TARGET


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()

CAT = list( set(X_train.columns) & set(loader.category()) )

#dtrain = lgb.Dataset(X_train, y_train, 
#                     categorical_feature=CAT)
COL = X_train.columns.tolist()

#X_train.head().to_csv(SUBMIT_FILE_PATH.replace('.csv', '_X.csv'),
#                         index=False, compression='gzip')

X_test = loader.test()[COL]

# =============================================================================
# def
# =============================================================================
file_psuedo = glob('../psuedo/*.f')
def get_psuedo(fileno=None):
    if fileno is None:
        file = np.random.choice(file_psuedo)
    else:
        file = file_psuedo[fileno]
    return pd.read_feather(file)

def get_dtrain(fileno=None):
    df = get_psuedo(fileno)
    X_psu = df.drop('TARGET', axis=1)
    y_psu = df['TARGET']
    X_train2 = pd.concat([X_train, X_psu], ignore_index=True)
    y_train2 = pd.concat([y_train, y_psu], ignore_index=True)
    dtrain = lgb.Dataset(X_train2, y_train2, free_raw_data=False,
                         categorical_feature=CAT)
    return dtrain

dtrains = [get_dtrain(i) for i in range(10)]
# =============================================================================
# training
# =============================================================================
models = []
for i in range(LOOP):
    print(f'LOOP: {i}')
    model = None
    for j in range(NUM_CHANGE_DATA):
        gc.collect()
        param.update({'seed':np.random.randint(9999)})
        model = lgb.train(param, dtrains[j%10], NROUND, init_model=model,
                          categorical_feature=CAT)
    models.append(model)


gc.collect()

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






