#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:39:16 2018

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
from sklearn.model_selection import GroupKFold
import count
import utils_cat
import utils
utils.start(__file__)
#==============================================================================

NFOLD = 5

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

use_files = []


#os.system(f'rm -rf ../feature_prev_unused')
#os.system(f'mkdir ../feature_prev_unused')

# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/prev_label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

sub_train = utils.read_pickles('../data/prev_train', ['SK_ID_CURR']).set_index('SK_ID_CURR')
sub_train['y'] = y.values
sub_train['cnt'] = sub_train.index.value_counts()
sub_train['w'] = 1 / sub_train.cnt.values

group_kfold = GroupKFold(n_splits=NFOLD)
sub_train['g'] = sub_train.index % NFOLD

CAT = list( set(X.columns)&set(utils_cat.ALL))

# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
model = lgb.train(param, dtrain, 3000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])

imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)


def multi_touch(arg):
    os.system(f'touch "../feature_prev_unused/{arg}.f"')


col = imp[imp['split']==0]['feature'].tolist()
pool = Pool(cpu_count())
pool.map(multi_touch, col)
pool.close()

#==============================================================================
utils.end(__file__)
#utils.stop_instance()

