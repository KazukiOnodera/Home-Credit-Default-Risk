#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:45:39 2018

@author: kazuki.onodera
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
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)

NFOLD = 5

LOOP = 10

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

imp_files = ['LOG/imp_181_imp.py.csv', 
             'LOG/imp_182_imp.py.csv', 
             'LOG/imp_183_imp.py.csv', 
             'LOG/imp_184_imp.py.csv', 
             'LOG/imp_185_imp.py.csv', 
             'LOG/imp_790_imp.py.csv', ]


adv = pd.read_feather('../feature/train_f190_adv.f')

# =============================================================================
# all data
# =============================================================================

files_tr = []

for p in imp_files:
    imp = pd.read_csv(p)
    imp = imp[imp.split>2]
    files_tr += ('../feature/train_' + imp.feature + '.f').tolist()

files_tr = sorted(set(files_tr))

print('features:', len(files_tr))

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

#X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')

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



model_all = []
for i in range(LOOP):
    ix = (adv.f190_adv*2) > np.random.uniform(size=adv.shape[0])
    dtrain = lgb.Dataset(X.loc[ix], y.loc[ix], categorical_feature=CAT )
    gc.collect()
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    print(i, ret['auc-mean'][-1])


imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)


#==============================================================================
utils.end(__file__)
utils.stop_instance()




