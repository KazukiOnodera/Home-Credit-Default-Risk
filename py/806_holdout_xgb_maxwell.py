#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:36:13 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import xgbextension as ex
import xgboost as xgb
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import train_test_split
#from glob import glob
import count
import utils_cat
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

params = {
          'booster': 'gbtree',  # gbtree, gblinear or dart
          'silent': 1,  # 0:printing mode 1:silent mode.
          'nthread': cpu_count(),
          'eta': 0.01,
          'gamma': 0.1,
          'max_depth': 6,
          'min_child_weight': 100,
          # 'max_delta_step': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.8,
          'lambda': 0.1,  # L2 regularization term on weights.
          'alpha': 0.1,  # L1 regularization term on weights.
          'tree_method': 'hist',
          # 'sketch_eps': 0.03,
          'scale_pos_weight': 1,
          # 'updater': 'grow_colmaker,prune',
          # 'refresh_leaf': 1,
          # 'process_type': 'default',
          'grow_policy': 'depthwise',
          # 'max_leaves': 0,
          'max_bin': 256,
          # 'predictor': 'cpu_predictor',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
#          'seed': SEED
          }


use_files = []


# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1); gc.collect()
y = utils.read_pickles('../data/label').TARGET

maxwell = pd.read_feather('../feature_someone/Maxwell_train.f')
X = pd.concat([X, maxwell], axis=1); gc.collect()


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'category: {CAT}')

X = pd.get_dummies(X, columns=CAT, drop_first=True)

# =============================================================================
# holdout
# =============================================================================
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, 
                                                    random_state=SEED,  test_size=0.2)


del X, y; gc.collect()



# =============================================================================
# training
# =============================================================================
dtrain = xgb.DMatrix(X_train, y_train); del X_train, y_train; gc.collect()
dvalid = xgb.DMatrix(X_valid, y_valid); del X_valid, y_valid; gc.collect()

watchlist = [(dtrain, 'train'),(dvalid, 'valid')]

model = xgb.train(params, dtrain, 9999, watchlist, verbose_eval=10,
                  early_stopping_rounds=50)



result = f"CV valid-auc: { model.best_score }"
print(result)

utils.send_line(result)


# =============================================================================
# imp
# =============================================================================
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


#col = imp[imp['split']==0]['feature'].tolist()
#for c in col:
#    os.system(f'touch "../unused_feature/{c}.f"')

# =============================================================================
# 
# =============================================================================
#col = imp['index'][:20].tolist()
#dtrain = lgb.Dataset(X[col], y, categorical_feature=list( set(col)&set(categorical_feature)) )
#gc.collect()
#
#ret = lgb.cv(param, dtrain, 9999, nfold=5,
#             early_stopping_rounds=50, verbose_eval=10,
#             seed=SEED)
#
#result = f"CV auc-mean(20 features) {ret['auc-mean'][-1]}"
#print(result)
#utils.send_line(result)


#==============================================================================
utils.end(__file__)

