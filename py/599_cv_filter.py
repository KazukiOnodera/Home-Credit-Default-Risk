#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:49:36 2018

@author: Kazuki
"""

from glob import glob
from os import system
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import multiprocessing
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

base_folder = ['../data/001_train']
target_folders = glob('../data/5*_train')
folders = base_folder+target_folders

X = pd.concat([
               utils.read_pickles(f) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 255,
         'max_bin': 255,
         'colsample_bytree': 0.1,
         'subsample': 0.5,
         'nthread': int(multiprocessing.cpu_count()/2),
#         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
         
         'seed': SEED
         }


categorical_feature = ['NAME_CONTRACT_TYPE',
                     'CODE_GENDER',
                     'FLAG_OWN_CAR',
                     'FLAG_OWN_REALTY',
                     'NAME_TYPE_SUITE',
                     'NAME_INCOME_TYPE',
                     'NAME_EDUCATION_TYPE',
                     'NAME_FAMILY_STATUS',
                     'NAME_HOUSING_TYPE',
                     'OCCUPATION_TYPE',
                     'WEEKDAY_APPR_PROCESS_START',
                     'ORGANIZATION_TYPE',
                     'FONDKAPREMONT_MODE',
                     'HOUSETYPE_MODE',
                     'WALLSMATERIAL_MODE',
                     'EMERGENCYSTATE_MODE']

categorical_feature += ['prev_DL1-0NAME_CONTRACT_TYPE', 'prev_DL1-0WEEKDAY_APPR_PROCESS_START', 'prev_DL1-0NAME_CASH_LOAN_PURPOSE', 'prev_DL1-0NAME_CONTRACT_STATUS', 'prev_DL1-0NAME_PAYMENT_TYPE', 'prev_DL1-0CODE_REJECT_REASON', 'prev_DL1-0NAME_TYPE_SUITE', 'prev_DL1-0NAME_CLIENT_TYPE', 'prev_DL1-0NAME_GOODS_CATEGORY', 'prev_DL1-0NAME_PORTFOLIO', 'prev_DL1-0NAME_PRODUCT_TYPE', 'prev_DL1-0CHANNEL_TYPE', 'prev_DL1-0NAME_SELLER_INDUSTRY', 'prev_DL1-0NAME_YIELD_GROUP', 'prev_DL1-0PRODUCT_COMBINATION', 'prev_DL1-1NAME_CONTRACT_TYPE', 'prev_DL1-1WEEKDAY_APPR_PROCESS_START', 'prev_DL1-1NAME_CASH_LOAN_PURPOSE', 'prev_DL1-1NAME_CONTRACT_STATUS', 'prev_DL1-1NAME_PAYMENT_TYPE', 'prev_DL1-1CODE_REJECT_REASON', 'prev_DL1-1NAME_TYPE_SUITE', 'prev_DL1-1NAME_CLIENT_TYPE', 'prev_DL1-1NAME_GOODS_CATEGORY', 'prev_DL1-1NAME_PORTFOLIO', 'prev_DL1-1NAME_PRODUCT_TYPE', 'prev_DL1-1CHANNEL_TYPE', 'prev_DL1-1NAME_SELLER_INDUSTRY', 'prev_DL1-1NAME_YIELD_GROUP', 'prev_DL1-1PRODUCT_COMBINATION', 'prev_DL1-2NAME_CONTRACT_TYPE', 'prev_DL1-2WEEKDAY_APPR_PROCESS_START', 'prev_DL1-2NAME_CASH_LOAN_PURPOSE', 'prev_DL1-2NAME_CONTRACT_STATUS', 'prev_DL1-2NAME_PAYMENT_TYPE', 'prev_DL1-2CODE_REJECT_REASON', 'prev_DL1-2NAME_TYPE_SUITE', 'prev_DL1-2NAME_CLIENT_TYPE', 'prev_DL1-2NAME_GOODS_CATEGORY', 'prev_DL1-2NAME_PORTFOLIO', 'prev_DL1-2NAME_PRODUCT_TYPE', 'prev_DL1-2CHANNEL_TYPE', 'prev_DL1-2NAME_SELLER_INDUSTRY', 'prev_DL1-2NAME_YIELD_GROUP', 'prev_DL1-2PRODUCT_COMBINATION']

dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
#dtrain.construct()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
#             categorical_feature=list( set(X.columns)&set(categorical_feature)),
             seed=SEED)
print(f"CV auc-mean {ret['auc-mean'][-1]}")

dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
model = lgb.train(param, dtrain, len(ret['auc-mean']))
#model = lgb.train(param, dtrain, 300, valid_sets=[dtrain], valid_names=['train'])

imp = ex.getImp(model)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)
"""

imp = pd.read_csv('LOG/imp_111-1_cv_filter.py.csv')


"""

# =============================================================================
# 
# =============================================================================
imp = imp.set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()

import gc

def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        print(folder)
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        del df; gc.collect()
        
        folder = folder.replace('_train', '_test')
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        
    else:
        print(f'{folder} doesnt have valid features')
        pass
    

[read_pickle(f, feature_all) for f in target_folders]

#==============================================================================
utils.end(__file__)

