#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:47:14 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f101_'


os.system(f'rm ../feature_bureau/t*_{PREF}*')

# =============================================================================
# 
# =============================================================================

train = utils.read_pickles('../data/bureau_train').drop(['SK_ID_CURR', 'SK_ID_BUREAU', 'TARGET'], axis=1)
test  = utils.read_pickles('../data/bureau_test').drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis=1)

categorical_features = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

le = LabelEncoder()
for c in categorical_features:
    train[c].fillna('na dayo', inplace=True)
    test[c].fillna('na dayo', inplace=True)
    le.fit( train[c].append(test[c]) )
    train[c] = le.transform(train[c])
    test[c]  = le.transform(test[c])

utils.to_feature(train.add_prefix(PREF), '../feature_bureau/train')
utils.to_feature(test.add_prefix(PREF),  '../feature_bureau/test')

#==============================================================================
utils.end(__file__)



