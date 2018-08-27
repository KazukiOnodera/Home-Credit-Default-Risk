#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:50:13 2018

@author: kazuki.onodera


latest previous application

"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f158_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# load
# =============================================================================
prev = utils.read_pickles('../data/future_application')
#base = prev[[KEY]].drop_duplicates().set_index(KEY)

# latest
prev_l = prev.sort_values([KEY, 'DAYS_DECISION'], ascending=[True, False]).drop_duplicates(KEY, keep='first').reset_index(drop=True)


# =============================================================================
# label encoding
# =============================================================================
categorical_features = prev_l.select_dtypes('O').columns.tolist()
"""

['NAME_CONTRACT_TYPE',
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

['f158_NAME_CONTRACT_TYPE',
 'f158_WEEKDAY_APPR_PROCESS_START',
 'f158_NAME_CASH_LOAN_PURPOSE',
 'f158_NAME_CONTRACT_STATUS',
 'f158_NAME_PAYMENT_TYPE',
 'f158_CODE_REJECT_REASON',
 'f158_NAME_TYPE_SUITE',
 'f158_NAME_CLIENT_TYPE',
 'f158_NAME_GOODS_CATEGORY',
 'f158_NAME_PORTFOLIO',
 'f158_NAME_PRODUCT_TYPE',
 'f158_CHANNEL_TYPE',
 'f158_NAME_SELLER_INDUSTRY',
 'f158_NAME_YIELD_GROUP',
 'f158_PRODUCT_COMBINATION']

"""

le = LabelEncoder()
for c in categorical_features:
    prev_l[c].fillna('na dayo', inplace=True)
    prev_l[c] = le.fit_transform(prev_l[c])




# =============================================================================
# merge
# =============================================================================
train = utils.load_train([KEY])
test = utils.load_test([KEY])

train2 = pd.merge(train, prev_l, on=KEY, how='left').drop([KEY, 'SK_ID_PREV'], axis=1)

test2 = pd.merge(test, prev_l, on=KEY, how='left').drop([KEY, 'SK_ID_PREV'], axis=1)

#utils.remove_feature(train2)
#train2, test2 = train2.align(test2, join='inner', axis=1)

# =============================================================================
# output
# =============================================================================
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)


