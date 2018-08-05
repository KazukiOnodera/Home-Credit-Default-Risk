#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:20:04 2018

@author: Kazuki
"""


import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import xgbextension as ex
import xgboost as xgb
from multiprocessing import cpu_count, Pool
from collections import defaultdict
#from glob import glob
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

HEAD = 1000 * 100
#HEAD = None

RESET = False

ONLY_ME = True

EXE_802 = True


params = {
          'booster': 'gbtree',  # gbtree, gblinear or dart
          'silent': 1,  # 0:printing mode 1:silent mode.
          'nthread': cpu_count(),
          'eta': 0.01,
          'gamma': 0.1,
          'max_depth': 6,
          'min_child_weight': 100,
          # 'max_delta_step': 0,
          'subsample': 0.6,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.5,
          'lambda': 0.1,  # L2 regularization term on weights.
          'alpha': 0.1,  # L1 regularization term on weights.
          'tree_method': 'auto',
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
          'tree_method':'hist',
          'eval_metric': 'auc',
          'seed': SEED
          }


if ONLY_ME:
    use_files = ['train_f']
else:
    use_files = ['train_']


# =============================================================================
# reset load
# =============================================================================
#if RESET:
#    os.system(f'rm -rf ../feature_unused')
#    os.system(f'mkdir ../feature_unused')
#
#    files = utils.get_use_files(use_files, True)
#    
#    if HEAD is not None:
#        X = pd.concat([
#                        pd.read_feather(f).head(HEAD) for f in tqdm(files, mininterval=60)
#                       ], axis=1)
#        y = utils.read_pickles('../data/label').head(HEAD).TARGET
#    else:
#        X = pd.concat([
#                        pd.read_feather(f) for f in tqdm(files, mininterval=60)
#                       ], axis=1)
#        y = utils.read_pickles('../data/label').TARGET
#    
#    
#    if X.columns.duplicated().sum()>0:
#        raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
#    print('no dup :) ')
#    print(f'X.shape {X.shape}')
#    
#    gc.collect()
#    
#    CAT = list( set(X.columns)&set(utils_cat.ALL))
#    print(f'CAT: {CAT}')
#    
#    # =============================================================================
#    # imp
#    # =============================================================================
#    dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
#    model = lgb.train(param, dtrain, 1000)
#    imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])
#    
#    
#    imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)
#    
#    """
#    imp = pd.read_csv('LOG/imp_801_imp_lgb.py.csv')
#    """
#    
#    def multi_touch(arg):
#        os.system(f'touch "../feature_unused/{arg}.f"')
#    
#    
#    col = imp[imp['split']==0]['feature'].tolist()
#    pool = Pool(cpu_count())
#    pool.map(multi_touch, col)
#    pool.close()

# =============================================================================
# all data
# =============================================================================
files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'CAT: {CAT}')


X = pd.get_dummies(X, columns=CAT, drop_first=True)
gc.collect()

col_before = X.columns.tolist()

#X.columns = [c.replace(',', '') for c in X.columns]
X.columns = [f'f{i}' for i,e in enumerate(X.columns)]

col_after = X.columns.tolist()

col_di = dict(zip(col_after, col_before))
## =============================================================================
## train
## =============================================================================
#dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
#model = lgb.train(param, dtrain, 3000)
#imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])

# =============================================================================
# cv
# =============================================================================
dtrain = xgb.DMatrix(X, y)
gc.collect()

model = xgb.train(params, dtrain, 3000)

imp = ex.getImp(model)
imp = imp.replace(col_di)
for c in CAT:
    imp.loc[imp.feature.str.startswith(c), 'feature'] = c

imp = imp.groupby('feature').sum().reset_index()

for c in ['weight', 'gain', 'cover']:
    imp[c] /= imp[c].max()

imp['total'] = imp.sum(1)
imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

imp.to_csv(f'LOG/imp_{__file__}-2.csv', index=False)




#def multi_touch(arg):
#    os.system(f'touch "../feature_unused/{arg}.f"')
#
#col = imp[imp['split']==0]['feature'].tolist()
#pool = Pool(cpu_count())
#pool.map(multi_touch, col)
#pool.close()

if EXE_802:
    os.system(f'nohup python -u 802_cv_lgb.py > LOG/log_802_cv_lgb.py.txt &')

#==============================================================================
utils.end(__file__)
#utils.stop_instance()

