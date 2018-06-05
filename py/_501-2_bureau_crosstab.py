#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:59:38 2018

@author: kazuki.onodera

bureau

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
PREF = 'bureau'
NTHREAD = 3


col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

col_group = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

# =============================================================================
# feature
# =============================================================================
bureau = utils.read_pickles('../data/bureau')

base = bureau[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(bureau[KEY], bureau[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(bureau[KEY], bureau[c1], normalize='index')
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

utils.to_pickles(train, '../data/501-2_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/501-2_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


