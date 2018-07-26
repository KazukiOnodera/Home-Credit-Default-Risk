#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 02:27:57 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
from sklearn.preprocessing import LabelEncoder
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f402_'

KEY = 'SK_ID_PREV'

month_start = -12*1 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature_prev/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance').drop('SK_ID_CURR', axis=1)
cre = cre[cre['MONTHS_BALANCE'].between(month_start, month_end)]

train = utils.read_pickles('../data/prev_train', [KEY])
test  = utils.read_pickles('../data/prev_test', [KEY])

le = LabelEncoder()
cre['NAME_CONTRACT_STATUS'] = le.fit_transform( cre['NAME_CONTRACT_STATUS'] )

col = [c for c in cre.columns if c.startswith('app_')]
cre.drop(col, axis=1, inplace=True)
# =============================================================================
# feature
# =============================================================================
df = pd.pivot_table(cre, index=KEY, columns='MONTHS_BALANCE')
df.columns = pd.Index([f'{e[0]}_{e[1]}' for e in df.columns.tolist()])
df.reset_index(inplace=True)

col = [c for c in df.columns if c.startswith('NAME_CONTRACT_STATUS')]
print(f'category: {col}')

# =============================================================================
# output
# =============================================================================
tmp = pd.merge(train, df, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature_prev/train')

tmp = pd.merge(test, df, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature_prev/test')

#==============================================================================
utils.end(__file__)



