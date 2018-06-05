#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:33:03 2018

@author: Kazuki
"""

import pandas as pd
import gc
from glob import glob
from tqdm import tqdm
import utils
utils.start(__file__)
#==============================================================================

KEY = 'SK_ID_CURR'
PREF = 'prev'

col_num = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 
           'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
           'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
           'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
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
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

base = prev[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(prev[KEY], prev[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(prev[KEY], prev[c1], normalize='index')
    df_norm.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
    df = pd.concat([df_sum, df_norm], axis=1)
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)


# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/101-2_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/101-2_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


