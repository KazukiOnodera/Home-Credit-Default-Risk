#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:16:22 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

# setting
day_start = -365*5 # min: -2922
day_end   = -365*4 # min: -2922

month_round = 1

PREF = 'ins_305_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
#ins = pd.read_csv('/Users/Kazuki/Home-Credit-Default-Risk/sample/sample_ins.csv')
ins = utils.read_pickles('../data/installments_payments')
#ins.drop('SK_ID_PREV', axis=1, inplace=True)
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]

# =============================================================================
# only forcus on delay
# =============================================================================
col_delayed = []
for i in range(0, 50, 5):
    ins[f'delayed_over{i}'] = (ins['days_delayed_payment']>i)*1
    col_delayed.append(f'delayed_over{i}')

gr = ins.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER'])
gr = gr[col_delayed].sum().groupby('SK_ID_CURR')

feature = pd.concat([
                     gr[col_delayed].min().add_suffix('_min'),
                     gr[col_delayed].mean().add_suffix('_mean'),
                     gr[col_delayed].max().add_suffix('_max'),
                     gr[col_delayed].std().add_suffix('_std'),
                     ], axis=1)

# =============================================================================
# 
# =============================================================================
1






feature.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])

test = utils.load_test([KEY])


train = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(train.add_prefix(PREF), '../feature/train')

test = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(test.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)

