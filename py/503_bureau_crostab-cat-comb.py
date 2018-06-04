#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:06:00 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
from itertools import combinations
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'bureau'
NTHREAD = 3


col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

col_cat_comb = list(combinations(col_cat, 2))

# =============================================================================
# feature
# =============================================================================
bureau = utils.read_pickles('../data/bureau')

base = bureau[[KEY]].drop_duplicates().set_index(KEY)

col_cat = []
for cc in col_cat_comb:
    c1, c2 = cc
    bureau[f'{c1}-{c2}'] = bureau[c1]+'-'+bureau[c2]
    col_cat.append(f'{c1}-{c2}')


# =============================================================================
# cat
# =============================================================================
li = []
col = []
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(bureau[KEY], bureau[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(bureau[KEY], bureau[c1], normalize='index')
    df_norm.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
    df = pd.concat([df_sum, df_norm], axis=1)
    li.append(df)
    col += df.columns.tolist()

base = pd.concat([base]+li, axis=1)
base[col] = base[col].fillna(-1)

# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/503_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/503_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


