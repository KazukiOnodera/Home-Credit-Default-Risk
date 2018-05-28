#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 02:40:30 2018

@author: Kazuki

previous_application

"""

import pandas as pd
import numpy as np
import gc
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'prev'
NTHREAD = 16

col_num = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 
           'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
           'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
           'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
           'CNT_PAYMENT',
           'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
           'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
           'NFLAG_INSURED_ON_APPROVAL']

col_cat = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']


# =============================================================================
# pivot
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

li = []
for cat in col_cat:
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_mean' for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num, aggfunc=np.sum)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_sum' for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num, aggfunc=np.std, fill_value=-1)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_std' for c in pt.columns]
    li.append(pt)

base = pd.concat(li).reset_index()

# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/201_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/201_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)



