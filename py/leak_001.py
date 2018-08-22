#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:49:32 2018

@author: kazuki.onodera
"""

import pandas as pd


# =============================================================================
# load
# =============================================================================
train = pd.read_csv('../input/application_train.csv.zip')
test  = pd.read_csv('../input/application_test.csv.zip')

trte = pd.concat([train, test], ignore_index=True)

prev = pd.read_csv('../input/previous_application.csv.zip')
prev = pd.merge(prev, trte[['SK_ID_CURR', 'CODE_GENDER']], on='SK_ID_CURR')


keys = ['AMT_ANNUITY',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE',
        'NAME_CONTRACT_TYPE',
        'WEEKDAY_APPR_PROCESS_START',
        'CODE_GENDER',
        'HOUR_APPR_PROCESS_START',
        'NAME_TYPE_SUITE']

prev_test = pd.merge(prev, test.drop_duplicates(keys), 
                     on=keys, how='inner')

# current best
sub = pd.read_csv('../data/stochastic_blending_v3-3_0.411rk7_0.304mw1_0.285t1_0.81031CV_0.3Adv.csv')

pos = pd.read_csv('../input/POS_CASH_balance.csv.zip')

# =============================================================================
# 
# =============================================================================
leak_ids = prev_test.SK_ID_CURR_y.value_counts()
leak_ids = leak_ids[leak_ids==1].index


pos = pos[pos.SK_ID_CURR.isin(leak_ids)]

dpd_mean = pos.groupby('SK_ID_CURR').SK_DPD.mean()

sub.sort_values('TARGET', inplace=True)

sub_0 = sub[sub.SK_ID_CURR.isin(dpd_mean[dpd_mean==0].index)]
sub_1 = sub[sub.SK_ID_CURR.isin(dpd_mean[dpd_mean>0].index)]








