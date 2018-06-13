#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:00:34 2018

@author: kazuki.onodera

days_weighted_delay

ins = pd.read_csv('/Users/Kazuki/Home-Credit-Default-Risk/py/sample_ins.csv')

"""

import os
import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
No = '305'
PREF = f'ins_{No}'


# =============================================================================
# load
# =============================================================================
ins = utils.read_pickles('../data/installments_payments')
ins.sort_values(['SK_ID_CURR', 'DAYS_ENTRY_PAYMENT', 'NUM_INSTALMENT_NUMBER'], 
                inplace=True, ascending=[True, False, False])

ins = utils.get_dummies(ins)
ins.drop('SK_ID_PREV', axis=1, inplace=True)

base = ins[[KEY]].drop_duplicates().set_index(KEY)

#ins['num'] = 1
#ins['num'] = ins.groupby(KEY)['num'].transform('count') - ins.groupby(KEY)['num'].cumsum()


# =============================================================================
# feature
# =============================================================================
feature1 = 'days_weighted_delay'
ins[feature1] = (ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']) * ins['days_delayed_payment']

features = [feature1]
for i in range(1, 11):
    decay = i * 0.0001 # decay rate per a day
    feature = f'days_weighted_delay_tsw{i}' # Time Series Weight
    ins[feature] = ins[feature1] * (1 + (ins['DAYS_ENTRY_PAYMENT']*decay) )
    features.append(feature)


gr = ins.groupby(KEY)
for feature in features:
    base[f'{PREF}_{feature}_min'] = gr[feature].min()
    base[f'{PREF}_{feature}_max'] = gr[feature].max()
    base[f'{PREF}_{feature}_mean'] = gr[feature].mean()
    base[f'{PREF}_{feature}_q25'] = gr[feature].quantile(0.25)
    base[f'{PREF}_{feature}_q50'] = gr[feature].quantile(0.5)
    base[f'{PREF}_{feature}_q75'] = gr[feature].quantile(0.75)






# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(train, '../feature/train')
del train; gc.collect()


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(test,  '../feature/test')






#==============================================================================
utils.end(__file__)


