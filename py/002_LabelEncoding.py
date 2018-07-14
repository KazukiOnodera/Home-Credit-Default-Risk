#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 02:31:35 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f002_'


os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# 
# =============================================================================

categorical_features = ['NAME_CONTRACT_TYPE',
#                         'CODE_GENDER',
#                         'FLAG_OWN_CAR',
#                         'FLAG_OWN_REALTY',
                         'NAME_TYPE_SUITE',
                         'NAME_INCOME_TYPE',
                         'NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS',
                         'NAME_HOUSING_TYPE',
                         'OCCUPATION_TYPE',
                         'WEEKDAY_APPR_PROCESS_START',
                         'ORGANIZATION_TYPE',
                         'FONDKAPREMONT_MODE',
                         'HOUSETYPE_MODE',
                         'WALLSMATERIAL_MODE',
#                         'EMERGENCYSTATE_MODE'
                         ]

train = utils.load_train(categorical_features)
test  = utils.load_test(categorical_features)

le = LabelEncoder()
for c in categorical_features:
    train[c].fillna('na dayo', inplace=True)
    test[c].fillna('na dayo', inplace=True)
    le.fit( train[c].append(test[c]) )
    train[c] = le.transform(train[c])
    test[c]  = le.transform(test[c])

utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)


