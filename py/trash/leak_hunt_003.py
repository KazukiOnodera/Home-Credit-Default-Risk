#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:51:31 2018

@author: Kazuki
"""

import pandas as pd

# =============================================================================

date_feature = [
                'DAYS_BIRTH',
                'DAYS_REGISTRATION',
                'DAYS_ID_PUBLISH',
                'DAYS_EMPLOYED'
                ]

other_feature = [
                'CODE_GENDER',
                'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE'
                ]

keys = date_feature + other_feature
# =============================================================================



test  = pd.read_csv('../input/application_test.csv.zip',
                    usecols=['SK_ID_CURR']+keys)


# new key
test['DAYS_BIRTH-m-DAYS_REGISTRATION'] = test['DAYS_BIRTH'] - test['DAYS_REGISTRATION']
test['DAYS_REGISTRATION-m-DAYS_ID_PUBLISH']   = test['DAYS_REGISTRATION'] - test['DAYS_ID_PUBLISH']
test['DAYS_ID_PUBLISH-m-DAYS_EMPLOYED']   = test['DAYS_ID_PUBLISH'] - test['DAYS_EMPLOYED']

keys2 = ['DAYS_BIRTH-m-DAYS_REGISTRATION', 'DAYS_REGISTRATION-m-DAYS_ID_PUBLISH',
         'DAYS_ID_PUBLISH-m-DAYS_EMPLOYED'] + other_feature
test[test.duplicated(keys2, False)].sort_values(keys2)




pd.concat([test[date_feature] + 1, test[other_feature]], axis=1)







