#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:31:33 2018

@author: kazuki.onodera
"""

from glob import glob
from os import system
import pandas as pd
import gc
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import multiprocessing
import utils
utils.start(__file__)
#==============================================================================

SEED = 71


FOLDERS = glob('../data/108*_train')
FOLDERS += glob('../data/109*_train')

X = pd.concat([
               utils.read_pickles(f) for f in (FOLDERS)
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
         'num_leaves': 511,
         'max_bin': 100,
         'colsample_bytree': 0.1,
         'subsample': 0.5,
         'nthread': 32,
         'bagging_freq': 1,
         'verbose': -1,
         
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

#dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
#
#ret = lgb.cv(param, dtrain, 9999, nfold=5,
#             early_stopping_rounds=50, verbose_eval=10,
#             seed=SEED)
#print(f"CV auc-mean {ret['auc-mean'][-1]}")

dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
#model = lgb.train(param, dtrain, len(ret['auc-mean']))
model = lgb.train(param, dtrain, 600)

imp = ex.getImp(model)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# =============================================================================
# 
# =============================================================================
imp = imp.set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()


def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    print(folder, len(col))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        del df; gc.collect()
        
        folder = folder.replace('_train', '_test')
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        
    else:
        print(f'{folder} doesnt have valid features')
        pass
    

[read_pickle(f, feature_all) for f in FOLDERS]

#==============================================================================
utils.end(__file__)



