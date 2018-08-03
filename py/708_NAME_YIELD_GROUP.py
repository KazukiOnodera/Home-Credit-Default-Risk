#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:32:44 2018

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
#from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from glob import glob
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f708_'

NFOLD = 5

SEED = 71

label_name = 'NAME_YIELD_GROUP'

param_mcl = {
         'objective': 'multiclass',
         'metric': 'multi_logloss',
         
         'learning_rate': 0.3,
         
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.9,
         'subsample': 0.9,
         'nthread': 32,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }


group_kfold = GroupKFold(n_splits=NFOLD)
np.random.seed(SEED)

os.system(f'rm ../feature/t*_{PREF}*')


# =============================================================================
# load
# =============================================================================
train = pd.read_csv('../input/application_train.csv.zip')
test = pd.read_csv('../input/application_test.csv.zip')
prev = pd.read_csv('../input/previous_application.csv.zip')

def mk_feature(df):
    df['AMT_CREDIT-d-AMT_ANNUITY']  = df['AMT_CREDIT'] / df['AMT_ANNUITY'] # how long should user pay?(month)
    df['AMT_GOODS_PRICE-d-AMT_ANNUITY'] = df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']# how long should user pay?(month)
    df['AMT_GOODS_PRICE-d-AMT_CREDIT']  = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['AMT_GOODS_PRICE-m-AMT_CREDIT']  = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    
    return

mk_feature(train)
mk_feature(test)
mk_feature(prev)

X_prev, X_train = prev.align(train, join='inner', axis=1)
X_train, X_test = X_train.align(test, join='inner', axis=1)


CAT = X_train.head().select_dtypes('O').columns.tolist()
le = LabelEncoder()
for c in CAT:
    X_prev[c].fillna('na dayo', inplace=True)
    X_train[c].fillna('na dayo', inplace=True)
    X_test[c].fillna('na dayo', inplace=True)
    le.fit( X_prev[c].append(X_train[c].append(X_test[c]) ))
    X_prev[c]  = le.transform(X_prev[c])
    X_train[c] = le.transform(X_train[c])
    X_test[c]  = le.transform(X_test[c])

# for multiclass
y_prev = pd.Series(le.fit_transform(prev[label_name]), name='y')


sub_prev = prev[['SK_ID_CURR']]
sub_prev['g'] = sub_prev.SK_ID_CURR % NFOLD

param_mcl['num_class'] = y_prev.nunique()
print('num_class:', param_mcl['num_class'])


sub_test = train[['SK_ID_CURR']]

X_prev.drop('SK_ID_CURR', axis=1, inplace=True)
X_train.drop('SK_ID_CURR', axis=1, inplace=True)
X_test.drop('SK_ID_CURR', axis=1, inplace=True)

# =============================================================================
# CV
# =============================================================================
dtrain = lgb.Dataset(X_prev, y_prev, 
                     categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param_mcl, dtrain, 99999, stratified=False,
                     folds=group_kfold.split(X_prev, y_prev, sub_prev['g']), 
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

imp = ex.getImp(models)

# =============================================================================
# predict
# =============================================================================

for i,model in enumerate(models):
    y_pred_train_ = model.predict(X_train)
    y_pred_test_  = model.predict(X_test)
    if i==0:
        y_pred_train = y_pred_train_
        y_pred_test  = y_pred_test_
    else:
        y_pred_train += y_pred_train_
        y_pred_test  += y_pred_test_

y_pred_train /= len(models)
y_pred_test /= len(models)


train = pd.DataFrame(y_pred_train).add_prefix(PREF)
test  = pd.DataFrame(y_pred_test).add_prefix(PREF)


# =============================================================================
# output
# =============================================================================
utils.to_feature(train, '../feature/train')
utils.to_feature(test, '../feature/test')




#==============================================================================
utils.end(__file__)


