#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 00:46:17 2018

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
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 3

NROUND = 4260

FEATURE_SIZE = 500

SUBMIT_FILE_PATH = '../output/725-4.csv.gz'

COMMENT = f'CV auc-mean(7 fold): 0.80387 + 0.00338 round: {NROUND} all(500)+nejumi'

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

# =============================================================================
# load train
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)

files = ('../feature/train_' + imp.head(FEATURE_SIZE).feature + '.f').tolist()
#files = utils.get_use_files(files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')
gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'category: {CAT}')

keys = sorted([c.split('_')[0] for c in X.columns])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')


dtrain = lgb.Dataset(X, y, 
                     categorical_feature=CAT)
COL = X.columns.tolist()

X.head().to_csv(SUBMIT_FILE_PATH.replace('.csv', '_X.csv'),
       index=False, compression='gzip')

del X, y; gc.collect()


# =============================================================================
# training
# =============================================================================
models = []
for i in range(LOOP):
    print(f'LOOP: {i}')
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
files = ('../feature/test_' + imp.head(FEATURE_SIZE).feature + '.f').tolist()

dtest = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)
dtest['nejumi'] = np.load('../feature_someone/test_nejumi.npy')
dtest = dtest[COL]

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


