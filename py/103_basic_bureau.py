#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:59:38 2018

@author: kazuki.onodera

bureau

"""

import numpy as np
import pandas as pd
import gc
import utils
#utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'bureau_'


col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['SK_ID_BUREAU', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']


# =============================================================================
# feature
# =============================================================================
bureau = utils.read_pickles('../data/bureau')

base = bureau[[KEY]].drop_duplicates().set_index(KEY)

gr = bureau.groupby(KEY)

def nunique(x):
    return len(set(x))

# stats
for c in col_num:
    gc.collect()
    print(c)
    base[f'{PREF}{c}_min'] = gr[c].min()
    base[f'{PREF}{c}_max'] = gr[c].max()
    base[f'{PREF}{c}_max-min'] = base[f'{PREF}{c}_max'] - base[f'{PREF}{c}_min']
    base[f'{PREF}{c}_mean'] = gr[c].mean()
    base[f'{PREF}{c}_std'] = gr[c].std()
    base[f'{PREF}{c}_sum'] = gr[c].sum()
    base[f'{PREF}{c}_nunique'] = gr[c].apply(nunique)
    

for c in col_cat:
    gc.collect()
    print(c)
    df = pd.crosstab(base[KEY], base[c])
    df.columns = ['{PREF}'+c.replace(' ', '-')+'_sum' for c in df.columns]
    base = pd.concat([base, df])

base.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/103_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/103_test',  utils.SPLIT_SIZE)






#==============================================================================
utils.end(__file__)


