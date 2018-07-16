#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 01:41:35 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f003_'

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

train = utils.load_train(categorical_features+['TARGET'])
test  = utils.load_test(categorical_features)

# =============================================================================
# 
# =============================================================================
usecols = []
for c in tqdm(categorical_features):
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
    
    usecols.append(c+'_ta')

# =============================================================================
# cardinality check
# =============================================================================
train['fold'] = 0
for i,(train_index, test_index) in enumerate(skf.split(train, train.TARGET)):
    train.loc[test_index, 'fold'] = i

for c in categorical_features:
    car_min = train.groupby(['fold', c]).size().min()
    print(f'{c}: {car_min}')


# =============================================================================
# output
# =============================================================================
utils.to_feature(train[usecols].add_prefix(PREF), '../feature/train')
utils.to_feature(test[usecols].add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

