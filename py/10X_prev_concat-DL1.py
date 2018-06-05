#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:43:41 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import utils
utils.start(__file__)
#==============================================================================
LOOP_ORDER = 3

KEY = 'SK_ID_CURR'


prev = utils.read_pickles('../data/previous_application')
prev.sort_values([KEY, 'DAYS_LAST_DUE_1ST_VERSION'], ascending=[True, False], inplace=True)
prev['order'] = 1
prev['order'] = prev.groupby(KEY)['order'].cumsum()
prev['order'] -= 1

col_cat = prev.select_dtypes('O').columns

for c in col_cat:
    prev[c].fillna('na dayo', inplace=True)
    prev[c] = le.fit_transform(prev[c])

base = prev[[KEY]].drop_duplicates().set_index(KEY)

col_cat_all = []
for i in range(LOOP_ORDER):
    prev_ = prev[prev['order']==i].drop_duplicates(KEY, keep='first')
    prev_ = prev_.set_index(KEY).add_prefix(f'prev_DL1-{i}')
    base = pd.concat([base, prev_], axis=1)
    col_cat_all += list( f'prev_DL1-{i}-' + col_cat )


# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(train, '../data/1XX_train', utils.SPLIT_SIZE)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(test,  '../data/1XX_test',  utils.SPLIT_SIZE)

print('col_cat_all:', col_cat_all)

"""

['prev_DL1-0NAME_CONTRACT_TYPE', 'prev_DL1-0WEEKDAY_APPR_PROCESS_START', 
'prev_DL1-0NAME_CASH_LOAN_PURPOSE', 'prev_DL1-0NAME_CONTRACT_STATUS', 
'prev_DL1-0NAME_PAYMENT_TYPE', 'prev_DL1-0CODE_REJECT_REASON', 
'prev_DL1-0NAME_TYPE_SUITE', 'prev_DL1-0NAME_CLIENT_TYPE', 
'prev_DL1-0NAME_GOODS_CATEGORY', 'prev_DL1-0NAME_PORTFOLIO', 
'prev_DL1-0NAME_PRODUCT_TYPE', 'prev_DL1-0CHANNEL_TYPE', 
'prev_DL1-0NAME_SELLER_INDUSTRY', 'prev_DL1-0NAME_YIELD_GROUP', 
'prev_DL1-0PRODUCT_COMBINATION', 'prev_DL1-1NAME_CONTRACT_TYPE', 
'prev_DL1-1WEEKDAY_APPR_PROCESS_START', 'prev_DL1-1NAME_CASH_LOAN_PURPOSE', 
'prev_DL1-1NAME_CONTRACT_STATUS', 'prev_DL1-1NAME_PAYMENT_TYPE',
 'prev_DL1-1CODE_REJECT_REASON', 'prev_DL1-1NAME_TYPE_SUITE', 
 'prev_DL1-1NAME_CLIENT_TYPE', 'prev_DL1-1NAME_GOODS_CATEGORY', 
 'prev_DL1-1NAME_PORTFOLIO', 'prev_DL1-1NAME_PRODUCT_TYPE', 
 'prev_DL1-1CHANNEL_TYPE', 'prev_DL1-1NAME_SELLER_INDUSTRY', 
 'prev_DL1-1NAME_YIELD_GROUP', 'prev_DL1-1PRODUCT_COMBINATION', 
 'prev_DL1-2NAME_CONTRACT_TYPE', 'prev_DL1-2WEEKDAY_APPR_PROCESS_START', 
 'prev_DL1-2NAME_CASH_LOAN_PURPOSE', 'prev_DL1-2NAME_CONTRACT_STATUS', 
 'prev_DL1-2NAME_PAYMENT_TYPE', 'prev_DL1-2CODE_REJECT_REASON', 
 'prev_DL1-2NAME_TYPE_SUITE', 'prev_DL1-2NAME_CLIENT_TYPE', 
 'prev_DL1-2NAME_GOODS_CATEGORY', 'prev_DL1-2NAME_PORTFOLIO', 
 'prev_DL1-2NAME_PRODUCT_TYPE', 'prev_DL1-2CHANNEL_TYPE', 
 'prev_DL1-2NAME_SELLER_INDUSTRY', 'prev_DL1-2NAME_YIELD_GROUP', 'prev_DL1-2PRODUCT_COMBINATION']

"""
#==============================================================================
utils.end(__file__)


