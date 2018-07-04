#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 13:51:18 2018

@author: Kazuki


今申し込んでいるローンは昔と比べるとどうか

"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f107_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
col = ['SK_ID_PREV',
       'SK_ID_CURR',
       'NAME_CONTRACT_TYPE',
       'AMT_ANNUITY',
     'AMT_APPLICATION',
     'AMT_CREDIT',
     'AMT_DOWN_PAYMENT',
     'AMT_GOODS_PRICE',
     'WEEKDAY_APPR_PROCESS_START',
     'HOUR_APPR_PROCESS_START',
     'FLAG_LAST_APPL_PER_CONTRACT',
     'NFLAG_LAST_APPL_IN_DAY',
     'RATE_DOWN_PAYMENT',
     'RATE_INTEREST_PRIMARY',
     'RATE_INTEREST_PRIVILEGED',
     'NAME_CASH_LOAN_PURPOSE',
     'NAME_CONTRACT_STATUS',
     'DAYS_DECISION',
     'NAME_PAYMENT_TYPE',
     'CODE_REJECT_REASON',
     'NAME_TYPE_SUITE',
     'NAME_CLIENT_TYPE',
     'NAME_GOODS_CATEGORY',
     'NAME_PORTFOLIO',
     'NAME_PRODUCT_TYPE',
     'CHANNEL_TYPE',
     'SELLERPLACE_AREA',
     'NAME_SELLER_INDUSTRY',
     'CNT_PAYMENT',
     'NAME_YIELD_GROUP',
     'PRODUCT_COMBINATION',
     'DAYS_FIRST_DRAWING',
     'DAYS_FIRST_DUE',
     'DAYS_LAST_DUE_1ST_VERSION',
     'DAYS_LAST_DUE',
     'DAYS_TERMINATION',
     'NFLAG_INSURED_ON_APPROVAL',
     
    'cnt_paid',
    'cnt_paid_ratio',
    'cnt_unpaid',
    'amt_paid',
    'amt_unpaid',
    'active',
    'completed',
     
     
     ]
prev = utils.read_pickles('../data/previous_application', col)
base = prev[[KEY]].drop_duplicates().set_index(KEY)
prev['is_approved'] = (prev['NAME_CONTRACT_STATUS']=='Approved')*1

col = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
rename_di = {#'NAME_CONTRACT_TYPE': 'app_NAME_CONTRACT_TYPE',
            'AMT_INCOME_TOTAL': 'app_AMT_INCOME_TOTAL', 
             'AMT_CREDIT': 'app_AMT_CREDIT', 
             'AMT_ANNUITY': 'app_AMT_ANNUITY',
             'AMT_GOODS_PRICE': 'app_AMT_GOODS_PRICE'}
trte = pd.concat([utils.load_train(col).rename(columns=rename_di), 
                  utils.load_test(col).rename(columns=rename_di)],
                  ignore_index=True)

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# feature
# =============================================================================

df = pd.merge(prev, trte, on=['SK_ID_CURR', 'NAME_CONTRACT_TYPE'], how='inner')
base['same_conttype_cnt'] = df.groupby('SK_ID_CURR').size()


# Approved
df = pd.merge(prev[prev['NAME_CONTRACT_STATUS']=='Approved'], trte, 
              on=['SK_ID_CURR', 'NAME_CONTRACT_TYPE'], how='inner')
base['same_conttype_approved_cnt'] = df.groupby('SK_ID_CURR').size()
base['same_conttype_active_ratio'] = df.groupby('SK_ID_CURR')['active'].mean()


# Refused
df = pd.merge(prev[prev['NAME_CONTRACT_STATUS']=='Refused'], trte, 
              on=['SK_ID_CURR', 'NAME_CONTRACT_TYPE'], how='inner')
base['same_conttype_refused_cnt'] = df.groupby('SK_ID_CURR').size()


# 'Approved', 'Refused'
df = pd.merge(prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])], trte, 
              on=['SK_ID_CURR', 'NAME_CONTRACT_TYPE'], how='inner')
base['same_conttype_appref_cnt'] = df.groupby('SK_ID_CURR').size()
base['same_conttype_approved_ratio'] = df.groupby('SK_ID_CURR')['is_approved'].mean()






base = base.fillna(0).reset_index()

# =============================================================================
# merge
# =============================================================================
train2 = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)

test2 = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.remove_feature(train2)
train2, test2 = train2.align(test2, join='inner', axis=1)

# =============================================================================
# output
# =============================================================================
#train2.drop(col_init, axis=1, inplace=True)
#test2.drop(col_init, axis=1, inplace=True)
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

