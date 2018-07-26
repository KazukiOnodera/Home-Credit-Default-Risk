#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:54:37 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
from sklearn.preprocessing import LabelEncoder
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f510_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
bure = utils.read_pickles('../data/bureau')

# first
bure = bure.sort_values([KEY, 'DAYS_CREDIT'], ascending=[True, False]).drop_duplicates(KEY, keep='last').reset_index(drop=True)

# =============================================================================
# label encoding
# =============================================================================
categorical_features = bure.select_dtypes('O').columns.tolist()

"""

['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

['f510_CREDIT_ACTIVE', 'f510_CREDIT_CURRENCY', 'f510_CREDIT_TYPE']

"""

le = LabelEncoder()
for c in categorical_features:
    bure[c].fillna('na dayo', inplace=True)
    bure[c] = le.fit_transform(bure[c])




# =============================================================================
# merge
# =============================================================================
train = utils.load_train([KEY])
test = utils.load_test([KEY])

train2 = pd.merge(train, bure, on=KEY, how='left').drop([KEY], axis=1)

test2 = pd.merge(test, bure, on=KEY, how='left').drop([KEY], axis=1)

#utils.remove_feature(train2)
#train2, test2 = train2.align(test2, join='inner', axis=1)

# =============================================================================
# output
# =============================================================================
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')



#==============================================================================
utils.end(__file__)


