#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 09:54:02 2018

@author: Kazuki
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
No = '505'
PREF = f'bureau_{No}'
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

bureau = bureau[bureau['DAYS_ENDDATE_FACT']>-365]
bureau = utils.get_dummies(bureau)

base = bureau[[KEY]].drop_duplicates().set_index(KEY)


def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================
gr = bureau.groupby(KEY)

# stats
base[f'{PREF}_{KEY}_size'] = gr.size()

base = pd.concat([
                base,
                gr.min().add_prefix(f'{PREF}_').add_suffix('_min'),
                gr.max().add_prefix(f'{PREF}_').add_suffix('_max'),
                gr.mean().add_prefix(f'{PREF}_').add_suffix('_mean'),
                gr.std().add_prefix(f'{PREF}_').add_suffix('_std'),
                gr.sum().add_prefix(f'{PREF}_').add_suffix('_sum'),
#                gr.median().add_prefix(f'{PREF}_').add_suffix('_median'),
                ], axis=1)



# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(train, f'../data/{No}_train', utils.SPLIT_SIZE)
del train; gc.collect()

test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(test,  f'../data/{No}_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


