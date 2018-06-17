#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:17:36 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================

PREF = 'oth_701'

KEY = 'SK_ID_CURR'


# =============================================================================
# 
# =============================================================================
col = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
       'AMT_CREDIT-dby-AMT_ANNUITY', 'DAYS_BIRTH']

train = utils.load_train([KEY]+col)
test = utils.load_test([KEY]+col)


col_init = train.columns.tolist()
# =============================================================================
# 
# =============================================================================

c1 = 'prev_101_total_debt_sum-p-app'
c2 = 'bure_501_AMT_CREDIT_SUM_sum'
train[f'{PREF}_total_debt-app-prev-bure'] = pd.read_feather(f'../feature/train_{c1}.f')[c1] + pd.read_feather(f'../feature/train_{c2}.f')[c2]
test[f'{PREF}_total_debt-app-prev-bure']  = pd.read_feather(f'../feature/test_{c1}.f')[c1]  + pd.read_feather(f'../feature/test_{c2}.f')[c2]

train[f'{PREF}_total_debt-app-prev-bure-dby-AMT_INCOME_TOTAL'] = train[f'{PREF}_total_debt-app-prev-bure'] / train['AMT_INCOME_TOTAL']
test[f'{PREF}_total_debt-app-prev-bure-dby-AMT_INCOME_TOTAL']  = test[f'{PREF}_total_debt-app-prev-bure']  / test['AMT_INCOME_TOTAL']



# =============================================================================
# output
# =============================================================================
train.drop(col_init, axis=1, inplace=True)
test.drop(col_init, axis=1, inplace=True)

utils.to_feature(train, '../feature/train')
utils.to_feature(test,  '../feature/test')



#==============================================================================
utils.end(__file__)


