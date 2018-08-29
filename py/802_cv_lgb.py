#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 01:29:16 2018

@author: Kazuki

nohup python run2.py 801_imp_lgb.py 802_cv_lgb.py &

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
from glob import glob
from sklearn.model_selection import GroupKFold
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)

NFOLD = 7

HEADS = list(range(500, 2300, 100))

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
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')

imp.sort_values('total', ascending=False, inplace=True)

y = utils.read_pickles('../data/label').TARGET

drop_ids = pd.read_csv('../data/drop_ids.csv')['SK_ID_CURR']
SK_ID_CURR = utils.load_train(['SK_ID_CURR'])

# =============================================================================
# groupKfold
# =============================================================================
#sk_tbl = pd.read_csv('../data/user_id_v7.csv.gz') # TODO: check
#user_tbl = sk_tbl.user_id.drop_duplicates().reset_index(drop=True).to_frame()
#
#sub_train = pd.read_csv('../input/application_train.csv.zip', usecols=['SK_ID_CURR']).set_index('SK_ID_CURR')
#sub_train['y'] = y.values
#
#group_kfold = GroupKFold(n_splits=NFOLD)

# =============================================================================
# shuffle fold
# =============================================================================
#ids = list(range(user_tbl.shape[0]))
#np.random.shuffle(ids)
#user_tbl['g'] = np.array(ids) % NFOLD
#sk_tbl_ = pd.merge(sk_tbl, user_tbl, on='user_id', how='left').set_index('SK_ID_CURR')
#
#sub_train['g'] = sk_tbl_.g

for HEAD in HEADS:
    files = ('../feature/train_' + imp.head(HEAD).feature + '.f').tolist()
    
#    files = utils.get_use_files(use_files, True)
    
    X = pd.concat([
                    pd.read_feather(f) for f in tqdm(files, mininterval=60)
                   ], axis=1)
    
    # =============================================================================
    # remove old users
    # =============================================================================
    X['SK_ID_CURR'] = SK_ID_CURR
    
    y = y[~X.SK_ID_CURR.isin(drop_ids)]
    X = X[~X.SK_ID_CURR.isin(drop_ids)].drop('SK_ID_CURR', axis=1)
    
    if X.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X.shape {X.shape}')
    
    gc.collect()
    
    CAT = list( set(X.columns)&set(utils_cat.ALL))
#    folds = group_kfold.split(X, sub_train['y'], sub_train['g'])
    
    # =============================================================================
    # cv
    # =============================================================================
    dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
    gc.collect()
    
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    result = f"CV auc-mean({SEED}:{HEAD}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
    print(result)
    
    utils.send_line(result)



#==============================================================================
utils.end(__file__)
#utils.stop_instance()


