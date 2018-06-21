#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:53:01 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

# setting
day_start = -365*2 # min: -2922
day_end   = -365   # min: -2922

month_round = 1

PREF = 'ins_306_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
#ins = pd.read_csv('/Users/Kazuki/Home-Credit-Default-Risk/sample/sample_ins_0.csv')
ins = utils.read_pickles('../data/installments_payments')
#ins.drop('SK_ID_PREV', axis=1, inplace=True)
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]


col_delayed_day = []
col_delayed_money = []
col_delayed_money_ratio = []
for i in range(0, 50, 5):
    c1 = f'delayed_day_{i}'
    ins[c1] = (ins['days_delayed_payment']>i)*1
    col_delayed_day.append(c1)
    
    c2 = f'delayed_money_{i}'
    ins[c2] = ins[c1] * ins.AMT_PAYMENT
    col_delayed_money.append(c2)
    
    c3 = f'delayed_money_ratio_{i}'
    ins[c3] = ins[c1] * ins.amt_ratio
    col_delayed_money_ratio.append(c3)


col_not_delayed_day = []
col_not_delayed_money = []
col_not_delayed_money_ratio = []
for i in range(0, 50, 5):
    c1 = f'not-delayed_day_{i}'
    ins[c1] = (ins['days_delayed_payment']<=i)*1
    col_not_delayed_day.append(c1)
    
    c2 = f'not-delayed_money_{i}'
    ins[c2] = ins[c1] * ins.AMT_PAYMENT
    col_not_delayed_money.append(c2)
    
    c3 = f'not-delayed_money_ratio_{i}'
    ins[c3] = ins[c1] * ins.amt_ratio
    col_not_delayed_money_ratio.append(c3)

gr1 = ins.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER'])

# =============================================================================
# features
# =============================================================================

def mk_feature(col):
    
    gr2 = gr1[col].sum().groupby('SK_ID_CURR')
    
    feature = pd.concat([
                         gr2.min().add_suffix('_min'),
                         gr2.mean().add_suffix('_mean'),
                         gr2.max().add_suffix('_max'),
                         gr2.std().add_suffix('_std'),
#                         gr2.sum().add_suffix('_sum'),
                         ], axis=1)
    return feature

col_list = [col_delayed_day, col_delayed_money, col_delayed_money_ratio,
            col_not_delayed_day, col_not_delayed_money, col_not_delayed_money_ratio]

feature = [mk_feature(col) for col in col_list]

feature = pd.concat(feature, axis=1)


utils.remove_feature(feature)
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

