#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:51:31 2018

@author: Kazuki
"""

import pandas as pd

DATE_FROM = -50
DATE_TO = 50

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

date_feature = [
                'DAYS_BIRTH',
                'DAYS_REGISTRATION',
                'DAYS_ID_PUBLISH',
                'DAYS_EMPLOYED'
                ]


test  = pd.read_csv('../input/application_test.csv.zip',
                    usecols=['SK_ID_CURR']+keys)



test[date_feature] + 1







