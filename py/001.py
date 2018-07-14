#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:24:47 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f001_'


os.system(f'rm ../feature/t*_{PREF}*')

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

# =============================================================================
# 
# =============================================================================
train = utils.load_train().drop(['SK_ID_CURR', 'TARGET']+categorical_features, axis=1)
test  = utils.load_test().drop(['SK_ID_CURR']+categorical_features, axis=1)


utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

