#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 10:37:55 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f002_'


os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# load
# =============================================================================

cat = ['NAME_CONTRACT_TYPE',
     'CODE_GENDER',
     'FLAG_OWN_CAR',
     'FLAG_OWN_REALTY',
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
     'EMERGENCYSTATE_MODE']

train = utils.load_train(cat)
test  = utils.load_test(cat)

train_row = train.shape[0]

trte = pd.concat([train, test])

trte.fillna('na dayo', inplace=True)
trte = trte.astype(str)

cat_comb = list(combinations(cat, 2))

# =============================================================================
# concat
# =============================================================================
col = []
# 2
for c1,c2 in cat_comb:
    trte[f'{c1}-{c2}'] = trte[c1] + trte[c2]
    col.append( f'{c1}-{c2}' )

# 3
#for c1,c2,c3 in cat_comb:
#    trte[f'{c1}-{c2}-{c3}'] = trte[c1] + trte[c2] + trte[c3]
#    col.append( f'{c1}-{c2}-{c3}' )


print(col)


le = LabelEncoder()
for c in col:
    trte[c] = le.fit_transform(trte[c])



train = trte.iloc[:train_row][col]
test = trte.iloc[train_row:][col]

utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

