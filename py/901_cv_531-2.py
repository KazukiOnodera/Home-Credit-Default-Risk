#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:13:13 2018

@author: kazuki.onodera
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
system('rm SUCCESS_901-2')
#==============================================================================

SEED = 71
imp_name = 'LOG/imp_901_cv_531-1.py.csv'

n_usecols = 2000
# =============================================================================
# 
# =============================================================================

folders = ['../data/101_train'] + sorted(glob('../data/*_train_filtered'))

imp = pd.read_csv(imp_name).set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()[:n_usecols]


def read_pickles(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        return df
        
        
    else:
        print(f'{folder} doesnt have valid features')
        return pd.DataFrame()
    


X = pd.concat([
                read_pickles(f, feature_all) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

print(f'X.shape {X.shape}')

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 511,
         'max_bin': 1023,
         'colsample_bytree': 0.1,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
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

dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)))

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
             seed=SEED)
print(f"CV auc-mean {ret['auc-mean'][-1]}")

dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)))
model = lgb.train(param, dtrain, len(ret['auc-mean']))

imp = ex.getImp(model)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

system('touch SUCCESS_901-2')

#==============================================================================
utils.end(__file__)


