#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:18:26 2018

@author: kazuki.onodera
"""

#import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
import os
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'app_002_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
train = utils.load_train().drop(['TARGET'], axis=1)
test  = utils.load_test()#.drop(['SK_ID_CURR'], axis=1)

trte = pd.concat([train, test], ignore_index=True)

base = trte[[KEY]]#.set_index('SK_ID_CURR')


categorical_features = ['NAME_CONTRACT_TYPE',
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

money_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                  'AMT_GOODS_PRICE']


# =============================================================================
# category median(other) ratio
# =============================================================================
for cat in categorical_features:
    for mon in money_features:
        median_name = f'{cat}-{mon}_median'
        print( median_name )
        tbl = trte.groupby(cat)[mon].median().reset_index()
        tbl.columns = [cat, median_name]
        
        tmp = pd.merge(trte, tbl, on=cat, how='left')
        
        base[f'{median_name}_ratio'] = tmp[median_name] / tmp[mon]



# =============================================================================
# merge
# =============================================================================
train = utils.load_train([KEY])
test  = utils.load_test([KEY])


train2 = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
test2  = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)





utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

