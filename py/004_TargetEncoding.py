#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 03:53:07 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f004_'

FOLD = 5

SEED = 71

os.system(f'rm ../feature/t*_{PREF}*')

categorical_features = ['NAME_CONTRACT_TYPE',
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
                         ]


# =============================================================================
# 
# =============================================================================
skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)

train = utils.load_train(categorical_features+['TARGET']).fillna('na dayo')
test  = utils.load_test(categorical_features).fillna('na dayo')


col = []
cat_comb = list(combinations(categorical_features, 2))
for c1,c2 in cat_comb:
    train[f'{c1}-{c2}'] = train[c1] + train[c2]
    test[f'{c1}-{c2}'] = test[c1] + test[c2]
    col.append( f'{c1}-{c2}' )

# =============================================================================
# cardinality check
# =============================================================================
train['fold'] = 0
for i,(train_index, test_index) in enumerate(skf.split(train, train.TARGET)):
    train.loc[test_index, 'fold'] = i

for c in col:
    car_min = train.groupby(['fold', c]).size().min()
    print(f'{c}: {car_min}')

train.groupby(['fold', col[2]]).size()

# =============================================================================
# def
# =============================================================================
def multi(c):
    train[c+'_ta'] = 0
    for i,(train_index, test_index) in enumerate(skf.split(train, train.TARGET)):
        enc = train.iloc[train_index].groupby(c)['TARGET'].mean()
        train.set_index(c, inplace=True)
        train.iloc[test_index, -1] = enc
        train.reset_index(inplace=True)
    enc = train.groupby(c)['TARGET'].mean()
    test[c+'_ta'] = 0
    test.set_index(c, inplace=True)
    test.iloc[:,-1] = enc
    test.reset_index(inplace=True)
    
    utils.to_feature(train[[c+'_ta']].add_prefix(PREF), '../feature/train')
    utils.to_feature(test[[c+'_ta']].add_prefix(PREF),  '../feature/test')
# =============================================================================
# 
# =============================================================================

pool = Pool(10)
pool.map(multi, col)
pool.close()

#==============================================================================
utils.end(__file__)


