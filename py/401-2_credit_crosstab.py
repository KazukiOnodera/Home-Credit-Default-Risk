#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:35:23 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'cre'
NTHREAD = 3

col_num = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
           'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
           'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
           'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
           'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
           'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
           'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
           'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF']

col_cat = ['CNT_DRAWINGS_OTHER_CURRENT', 'NAME_CONTRACT_STATUS']

col_group = ['SK_ID_PREV', 'CNT_DRAWINGS_OTHER_CURRENT', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance')
base = cre[[KEY]].drop_duplicates().set_index(KEY)


def nunique(x):
    return len(set(x))

# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(cre[KEY], cre[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(cre[KEY], cre[c1], normalize='index')
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

utils.to_pickles(train, '../data/401-2_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/401-2_test',  utils.SPLIT_SIZE)




#==============================================================================
utils.end(__file__)


