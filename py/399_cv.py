#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:42:49 2018

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
import count
import utils
#utils.start(__file__)
#==============================================================================

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


# =============================================================================
# load
# =============================================================================

files = []
for i in range(318, 323):
    files += glob(f'../feature/train_f{i}*')

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

#COL = X.columns
#X.columns = ['f'+str(i) for i in range(X.shape[1])]
X['cat'] = 1
# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=['cat'])
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=7, categorical_feature=['cat'],
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(318~322): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)

# 0.783312

# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=['cat'])
model = lgb.train(param, dtrain, len(ret['auc-mean']))
#model = lgb.train(param, dtrain, 1000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

#def multi_touch(arg):
#    os.system(f'touch "../feature_unused/{arg}.f"')
#
#
#col = imp[imp['split']==0]['feature'].tolist()
#pool = Pool(cpu_count())
#pool.map(multi_touch, col)
#pool.close()



#==============================================================================
utils.end(__file__)

