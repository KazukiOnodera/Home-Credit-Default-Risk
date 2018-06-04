#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 00:52:14 2018

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
PREF = 'ins'
NTHREAD = 3

col_num = ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 
           'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']


col_group = ['SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']

# =============================================================================
# feature
# =============================================================================
ins = utils.read_pickles('../data/installments_payments')
base = ins[[KEY]].drop_duplicates().set_index(KEY)


def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================
gr = ins.groupby(KEY)

# stats
keyname = 'gby-'+KEY
base[f'{PREF}_{keyname}_size'] = gr.size()
for c in col_num:
    gc.collect()
    print(c)
    base[f'{PREF}_{keyname}_{c}_min'] = gr[c].min()
    base[f'{PREF}_{keyname}_{c}_max'] = gr[c].max()
    base[f'{PREF}_{keyname}_{c}_max-min'] = base[f'{PREF}_{keyname}_{c}_max'] - base[f'{PREF}_{keyname}_{c}_min']
    base[f'{PREF}_{keyname}_{c}_mean'] = gr[c].mean()
    base[f'{PREF}_{keyname}_{c}_std'] = gr[c].std()
    base[f'{PREF}_{keyname}_{c}_sum'] = gr[c].sum()
    base[f'{PREF}_{keyname}_{c}_nunique'] = gr[c].apply(nunique)

# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)

test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/105-1_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/105-1_test',  utils.SPLIT_SIZE)

os.system('rm ../data/tmp_ins*.p')





#==============================================================================
utils.end(__file__)


