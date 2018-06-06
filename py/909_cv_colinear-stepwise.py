#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 13:11:07 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
from time import sleep
import multiprocessing
import os
import gc
import utils
#utils.start(__file__)
#==============================================================================

SEED = 71

IMP_FNAME = 'LOG/imp_901_cv_527-1.py.csv'

THRESHOLD = 0.5

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
         'max_depth': -1,
         'num_leaves': 127,
         'max_bin': 127,
         'colsample_bytree': 0.5,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
         
         'seed': SEED, 
         'verbose': -1
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

# =============================================================================
# wait
# =============================================================================

#while True:
#    if os.path.isfile('SUCCESS_901'):
#        break
#    else:
#        sleep(60*1)
#
#utils.reset_time()

# =============================================================================
# 
# =============================================================================
imp = pd.read_csv(IMP_FNAME).set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()

folders = sorted(glob('../data/*_train'))

def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        return df
    else:
        print(f'{folder} desnt have valid columns')
        return pd.DataFrame()


X = pd.concat([
                read_pickle(f, feature_all) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


# =============================================================================
# LGB
# =============================================================================
use_features = feature_all[:]

# benchmark
dtrain = lgb.Dataset(X[use_features], y, 
                     categorical_feature=list( set(categorical_feature)&set(use_features) ))
ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=None,
             seed=SEED)

best_score = ret['auc-mean'][-1]
print(f'benchmark: {best_score}')
print(f'features: {use_features}')

corr = X.sample(99999).corr().abs()
col_del_candidates = []
for c in feature_all[:100]:
    col_del_candidates += list(corr[corr[c]>0.5][corr[c]<1].index)
col_del_candidates = pd.unique(col_del_candidates)

print(f'col_del_candidates: {len(col_del_candidates)} {col_del_candidates}')

for c in col_del_candidates:
    print()
    gc.collect()
    
    use_features_ = use_features[:]
    print(f'drop {c}')
    use_features_.remove(c)
    
    dtrain = lgb.Dataset(X[use_features_], y, 
                         categorical_feature=list( set(categorical_feature)&set(use_features_) ))
    ret = lgb.cv(param, dtrain, 9999, nfold=5,
                 early_stopping_rounds=50, verbose_eval=None,
                 seed=SEED)
    score = ret['auc-mean'][-1]
    print(f"auc-mean {score}")
    
    if best_score < score:
        print(f'UPDATE!    SCORE:{score:+.5f}    DIFF:{score-best_score:+.5f}')
        print(f'features: {use_features_}')
        best_score = score
        use_features = use_features_



#==============================================================================
utils.end(__file__)



