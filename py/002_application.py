#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 05:56:27 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
import utils
utils.start(__file__)
#==============================================================================

PREF = 'app_002_'

train = utils.load_train().drop(['SK_ID_CURR', 'TARGET'], axis=1)
test  = utils.load_test().drop(['SK_ID_CURR'], axis=1)

col_init = train.columns

df = pd.concat([ train, test], ignore_index=True)
# =============================================================================
# features
# =============================================================================
    
df['AMT_CREDIT-by-AMT_INCOME_TOTAL'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['AMT_INCOME_TOTAL-AMT_CREDIT']    = df['AMT_INCOME_TOTAL'] - df['AMT_CREDIT']

df['AMT_ANNUITY-by-AMT_INCOME_TOTAL'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['AMT_INCOME_TOTAL-AMT_ANNUITY']    = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']

df['AMT_GOODS_PRICE-by-AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
df['AMT_INCOME_TOTAL-AMT_GOODS_PRICE']    = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']

df.loc[df['CNT_CHILDREN']==0, 'CNT_CHILDREN'] = np.nan
df['AMT_INCOME_TOTAL-by-CNT_CHILDREN'] = df['AMT_INCOME_TOTAL']  / df['CNT_CHILDREN']
df['AMT_CREDIT-by-CNT_CHILDREN']       = df['AMT_CREDIT']        / df['CNT_CHILDREN']
df['AMT_ANNUITY-by-CNT_CHILDREN']      = df['AMT_ANNUITY']       / df['CNT_CHILDREN']
df['AMT_GOODS_PRICE-by-CNT_CHILDREN']  = df['AMT_GOODS_PRICE']   / df['CNT_CHILDREN']

df['EXT_SOURCE_1-by-AMT_INCOME_TOTAL'] = df['EXT_SOURCE_1'] / df['AMT_INCOME_TOTAL']
#df['EXT_SOURCE_2-by-AMT_INCOME_TOTAL'] = df['EXT_SOURCE_2'] / df['AMT_INCOME_TOTAL']
#df['EXT_SOURCE_3-by-AMT_INCOME_TOTAL'] = df['EXT_SOURCE_3'] / df['AMT_INCOME_TOTAL']

df.loc[df['DAYS_EMPLOYED']==365243, 'DAYS_EMPLOYED'] = np.nan
df['DAYS_EMPLOYED-DAYS_BIRTH']            = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
df['DAYS_REGISTRATION-DAYS_BIRTH']        = df['DAYS_REGISTRATION'] - df['DAYS_BIRTH']
df['DAYS_ID_PUBLISH-DAYS_BIRTH']          = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']
df['DAYS_LAST_PHONE_CHANGE-DAYS_BIRTH']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_BIRTH']

df['DAYS_REGISTRATION-DAYS_EMPLOYED']        = df['DAYS_REGISTRATION'] - df['DAYS_EMPLOYED']
df['DAYS_ID_PUBLISH-DAYS_EMPLOYED']          = df['DAYS_ID_PUBLISH'] - df['DAYS_EMPLOYED']
df['DAYS_LAST_PHONE_CHANGE-DAYS_EMPLOYED']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_EMPLOYED']

df['DAYS_ID_PUBLISH-DAYS_REGISTRATION']          = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
df['DAYS_LAST_PHONE_CHANGE-DAYS_REGISTRATION']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION']

df['DAYS_LAST_PHONE_CHANGE-DAYS_ID_PUBLISH']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_ID_PUBLISH']

#gr = df.groupby('REGION_POPULATION_RELATIVE')
#df['AMT_INCOME_TOTAL-rank_in-region'] = gr['AMT_INCOME_TOTAL'].rank(method='dense')
#df['AMT_CREDIT-rank_in-region']       = gr['AMT_CREDIT'].rank(method='dense')
#df['AMT_ANNUITY-rank_in-region'] = gr['AMT_ANNUITY'].rank(method='dense')
#df['AMT_GOODS_PRICE-rank_in-region'] = gr['AMT_GOODS_PRICE'].rank(method='dense')
#df['EXT_SOURCE_1-rank_in-region'] = gr['EXT_SOURCE_1'].rank(method='dense')
#df['EXT_SOURCE_2-rank_in-region'] = gr['EXT_SOURCE_2'].rank(method='dense')
#df['EXT_SOURCE_3-rank_in-region'] = gr['EXT_SOURCE_3'].rank(method='dense')
#df['CNT_CHILDREN-rank_in-region'] = gr['CNT_CHILDREN'].rank(method='dense')




df.drop(col_init, axis=1, inplace=True)

train = df.loc[:train.shape[0]-1].reset_index(drop=True)
test  = df.loc[train.shape[0]:].reset_index(drop=True)

# =============================================================================
# write
# =============================================================================
utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')



#==============================================================================
utils.end(__file__)

