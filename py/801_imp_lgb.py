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
from collections import defaultdict
#from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

HEAD = 1000 * 100
#HEAD = None

RESET = False

ONLY_ME = True

EXE_802 = True

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


if ONLY_ME:
    use_files = ['train_f']
else:
    use_files = ['train_']


# =============================================================================
# reset load
# =============================================================================
if RESET:
    os.system(f'rm -rf ../feature_unused')
    os.system(f'mkdir ../feature_unused')

    files = utils.get_use_files(use_files, True)
    
    if HEAD is not None:
        X = pd.concat([
                        pd.read_feather(f).head(HEAD) for f in tqdm(files, mininterval=60)
                       ], axis=1)
        y = utils.read_pickles('../data/label').head(HEAD).TARGET
    else:
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
    
    # =============================================================================
    # imp
    # =============================================================================
    dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
    model = lgb.train(param, dtrain, 1000)
    imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])
    
    
    imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)
    
    """
    imp = pd.read_csv('LOG/imp_801_imp_lgb.py.csv')
    """
    
    def multi_touch(arg):
        os.system(f'touch "../feature_unused/{arg}.f"')
    
    
    col = imp[imp['split']==0]['feature'].tolist()
    pool = Pool(cpu_count())
    pool.map(multi_touch, col)
    pool.close()

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

# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
model = lgb.train(param, dtrain, 3000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])

"""
imp[imp.feature.str.startswith('f312_')]

__file__ = '801_imp_lgb.py'
"""
imp.to_csv(f'LOG/imp_{__file__}-2.csv', index=False)


print('top100')
keys = sorted([c.split('_')[0] for c in imp.feature[:100]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top100 - 200')
keys = sorted([c.split('_')[0] for c in imp.feature[100:200]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top200 - 300')
keys = sorted([c.split('_')[0] for c in imp.feature[200:300]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top300 - 400')
keys = sorted([c.split('_')[0] for c in imp.feature[300:400]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top400 - 500')
keys = sorted([c.split('_')[0] for c in imp.feature[400:500]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top500 - 600')
keys = sorted([c.split('_')[0] for c in imp.feature[500:600]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')
    
print('top600 - 700')
keys = sorted([c.split('_')[0] for c in imp.feature[600:700]])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')



def multi_touch(arg):
    os.system(f'touch "../feature_unused/{arg}.f"')

col = imp[imp['split']==0]['feature'].tolist()
pool = Pool(cpu_count())
pool.map(multi_touch, col)
pool.close()

if EXE_802:
    os.system(f'nohup python -u 802_cv_lgb.py > LOG/log_802_cv_lgb.py.txt &')

#==============================================================================
utils.end(__file__)
#utils.stop_instance()

