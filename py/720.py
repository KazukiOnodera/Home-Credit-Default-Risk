#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:09:41 2018

@author: kazuki.onodera
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# load
# =============================================================================
usecols = ['AMT_ANNUITY',
            'AMT_CREDIT',
            'AMT_GOODS_PRICE',
            'NAME_CONTRACT_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'HOUR_APPR_PROCESS_START',
            'NAME_TYPE_SUITE']

categorical_features = ['NAME_CONTRACT_TYPE',
                         'NAME_TYPE_SUITE',
                         'WEEKDAY_APPR_PROCESS_START'
                         ]

train = pd.read_csv('../input/application_train.csv.zip', 
                    usecols=['SK_ID_CURR', 'CODE_GENDER']+usecols)
train['CODE_GENDER'] = 1 - (train['CODE_GENDER']=='F')*1

test  = pd.read_csv('../input/application_test.csv.zip', 
                    usecols=['SK_ID_CURR', 'CODE_GENDER']+usecols)
test['CODE_GENDER'] = 1 - (test['CODE_GENDER']=='F')*1

trte = pd.concat([train, test], ignore_index=True)

prev = pd.read_csv('../input/previous_application.csv.zip',
                   usecols=['SK_ID_PREV', 'SK_ID_CURR']+usecols)
prev = pd.merge(prev, trte[['SK_ID_CURR', 'CODE_GENDER']], on='SK_ID_CURR')


le = LabelEncoder()
for c in categorical_features:
    train[c].fillna('-NA-', inplace=True)
    test[c].fillna('-NA-', inplace=True)
    prev[c].fillna('-NA-', inplace=True)
    le.fit( train[c].append(test[c].append(prev[c])) )
    train[c] = le.transform(train[c])
    test[c]  = le.transform(test[c])
    prev[c]  = le.transform(prev[c])


# =============================================================================
# target
# =============================================================================
target = pd.read_csv('../input/POS_CASH_balance.csv.zip',
                     usecols=['SK_ID_PREV', 'SK_DPD']).groupby('SK_ID_PREV').SK_DPD.mean()








