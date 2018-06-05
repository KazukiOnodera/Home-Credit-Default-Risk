#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:03 2018

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
PREF = 'pos'

NTHREAD = 2

col_num = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
           'SK_DPD', 'SK_DPD_DEF']

col_cat = ['NAME_CONTRACT_STATUS']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
base = pos[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(pos[KEY], pos[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(pos[KEY], pos[c1], normalize='index')
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

utils.to_pickles(train, '../data/201-2_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/201-2_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


