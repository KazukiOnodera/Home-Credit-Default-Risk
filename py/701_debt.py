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
train = pd.read_feather('../feature/train_prev_101_debt_sum_rem-p-app.f')
test = pd.read_feather('../feature/test_prev_101_debt_sum_rem-p-app.f')

col_init = train.columns.tolist()
# =============================================================================
# 
# =============================================================================
train[f'{PREF}_total_debt-app-prev-bure'] = train['prev_101_debt_sum_rem-p-app'] + pd.read_feather('../feature/train_bure_501_AMT_CREDIT_SUM_DEBT_sum.f')['bure_501_AMT_CREDIT_SUM_DEBT_sum']
test[f'{PREF}_total_debt-app-prev-bure']  = test['prev_101_debt_sum_rem-p-app'] + pd.read_feather('../feature/test_bure_501_AMT_CREDIT_SUM_DEBT_sum.f')['bure_501_AMT_CREDIT_SUM_DEBT_sum']


# =============================================================================
# output
# =============================================================================
train.drop(col_init, axis=1, inplace=True)
test.drop(col_init, axis=1, inplace=True)

utils.to_feature(train, '../feature/train')
utils.to_feature(test,  '../feature/test')



#==============================================================================
utils.end(__file__)


