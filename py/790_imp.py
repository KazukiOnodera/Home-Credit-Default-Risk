#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:58:10 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
#from glob import glob
import count
import utils_cat
import utils
utils.start(__file__)
#==============================================================================

HEAD = 200000

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
         
         'colsample_bytree': 0.7,
         'subsample': 0.5,
         'nthread': 32,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }



use_files = ['train_f0', 'train_f2', 'train_f3', 'train_f4', 
             'train_f5', 'train_f6', 'train_f7']

#REMOVE_FEATURES = ['f023', 'f024']

# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

#tmp = []
#for f in files:
#    sw = False # skip switch
#    for r in REMOVE_FEATURES:
#        if r in f:
#            sw = True
#            break
#    if not sw:
#        tmp.append(f)
#
#files = tmp
print('features:', len(files))



X = pd.concat([
                pd.read_feather(f).head(HEAD) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').head(HEAD).TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

#X = X.rank(method='dense')
gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))

# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
#model = lgb.train(param, dtrain, len(ret['auc-mean']))
model = lgb.train(param, dtrain, 2000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

def multi_touch(arg):
    os.system(f'touch "../feature_unused/{arg}.f"')


#col = imp[imp['split']==0]['feature'].tolist()
#pool = Pool(cpu_count())
#pool.map(multi_touch, col)
#pool.close()


#==============================================================================
utils.end(__file__)



