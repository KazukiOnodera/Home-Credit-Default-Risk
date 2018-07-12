#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 04:00:57 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f101_'


os.system(f'rm ../feature_prev/t*_{PREF}*')

# =============================================================================
# 
# =============================================================================

train = utils.read_pickles('../data/prev_train').drop(['SK_ID_CURR', 'SK_ID_PREV', 'TARGET'], axis=1)
test  = utils.read_pickles('../data/prev_test').drop(['SK_ID_CURR', 'SK_ID_PREV'], axis=1)

categorical_features = ['NAME_CONTRACT_TYPE',
                         'WEEKDAY_APPR_PROCESS_START',
                         'NAME_CASH_LOAN_PURPOSE',
                         'NAME_CONTRACT_STATUS',
                         'NAME_PAYMENT_TYPE',
                         'CODE_REJECT_REASON',
                         'NAME_TYPE_SUITE',
                         'NAME_CLIENT_TYPE',
                         'NAME_GOODS_CATEGORY',
                         'NAME_PORTFOLIO',
                         'NAME_PRODUCT_TYPE',
                         'CHANNEL_TYPE',
                         'NAME_SELLER_INDUSTRY',
                         'NAME_YIELD_GROUP',
                         'PRODUCT_COMBINATION']

le = LabelEncoder()
for c in categorical_features:
    train[c].fillna('na dayo', inplace=True)
    test[c].fillna('na dayo', inplace=True)
    le.fit( train[c].append(test[c]) )
    train[c] = le.transform(train[c])
    test[c]  = le.transform(test[c])

utils.to_feature(train.add_prefix(PREF), '../feature_prev/train')
utils.to_feature(test.add_prefix(PREF),  '../feature_prev/test')

#==============================================================================
utils.end(__file__)


