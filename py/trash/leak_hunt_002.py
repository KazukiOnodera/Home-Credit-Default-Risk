#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 19:41:46 2018

@author: kazuki.onodera
"""

import pandas as pd

# =============================================================================
# load
# =============================================================================
train = pd.read_csv('../input/application_train.csv.zip')
test  = pd.read_csv('../input/application_test.csv.zip')

trte = pd.concat([train, test], ignore_index=True)

# =============================================================================
# train
# =============================================================================
keys = [
        'CODE_GENDER',
        'DAYS_BIRTH',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'NAME_EDUCATION_TYPE',
        'DAYS_EMPLOYED',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE'
        
        ]

keys += ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN']


tmp = train[train.duplicated(keys, False)].sort_values(keys)

tmp.to_csv('same_user.csv', index=False)



# =============================================================================
# test
# =============================================================================

keys = [
        'CODE_GENDER',
        'DAYS_BIRTH',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'NAME_EDUCATION_TYPE',
        'DAYS_EMPLOYED',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE'
        
        ]

keys += ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'REGION_POPULATION_RELATIVE']

test  = pd.read_csv('../input/application_test.csv.zip')

years = 1
for c in ['DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED',]:
    test[c] = (test[c] / (365*years)).astype(int)

tmp = test[test.duplicated(keys, False)].sort_values(keys)


tmp.to_csv('same_test_user_within1y.csv', index=False)


# =============================================================================
# trte
# =============================================================================
keys = [
        'CODE_GENDER',
        'DAYS_BIRTH',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        'NAME_EDUCATION_TYPE',
        'DAYS_EMPLOYED',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE'
        
        ]

keys += ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'REGION_POPULATION_RELATIVE']


tmp = trte[trte.duplicated(keys, False)].sort_values(keys)



