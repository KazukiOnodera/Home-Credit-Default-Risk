#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:39:16 2018

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
from multiprocessing import cpu_count, Pool
from collections import defaultdict
#from glob import glob
from sklearn.model_selection import GroupKFold
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)

HEAD = 1000 * 100

NFOLD = 4

LOOP = 3

RESET = False

ONLY_ME = False

EXE_802 = False

#REMOVE_FEATURES = ['f023', 'f024']

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

new_train_users = pd.read_csv('../data/new_train_users.csv').SK_ID_CURR

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
#files = utils.get_use_files(use_files, True)

#tmp = []
#for f in files:
#    sw = False # skip switch
#    for r in REMOVE_FEATURES:
#        if r in f:
#            sw = True
#            break
#    if not sw:
#        tmp.append(f)
#files = tmp

files = ('../feature/train_' + pd.read_csv('LOG/imp_remove-f15.csv').feature + '.f').tolist()

print('features:', len(files))

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
print(f'CAT: {CAT}')

# =============================================================================
# groupKfold
# =============================================================================
#sk_tbl = pd.read_csv('../data/user_id_v8.csv.gz') # TODO: check
#user_tbl = sk_tbl.user_id.drop_duplicates().reset_index(drop=True).to_frame()
#
#sub_train = pd.read_csv('../input/application_train.csv.zip', usecols=['SK_ID_CURR']).set_index('SK_ID_CURR')
#sub_train['y'] = y.values
#
#group_kfold = GroupKFold(n_splits=NFOLD)
#
#
## shuffle fold
#ids = list(range(user_tbl.shape[0]))
#np.random.shuffle(ids)
#user_tbl['g'] = np.array(ids) % NFOLD
#sk_tbl_ = pd.merge(sk_tbl, user_tbl, on='user_id', how='left').set_index('SK_ID_CURR')
#
#sub_train['g'] = sk_tbl_.g
#folds = group_kfold.split(X, sub_train['y'], sub_train['g'])


# =============================================================================
# remove old users
# =============================================================================
X = X[new_train_users]
y = y[new_train_users]

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

model_all = []
for i in range(LOOP):
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models

result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)
imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

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

