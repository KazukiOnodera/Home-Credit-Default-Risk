#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:56:34 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils
#utils.start(__file__)
#==============================================================================
PREF = 'bure_501_'

KEY = 'SK_ID_CURR'

# =============================================================================
# 
# =============================================================================
bure = utils.read_pickles('../data/bureau')
base = bure[[KEY]].drop_duplicates().set_index(KEY)

gr = bure.groupby(KEY)
gr_act = bure[bure['CREDIT_ACTIVE']=='Active'].groupby(KEY)
gr_cls = bure[bure['CREDIT_ACTIVE']=='Closed'].groupby(KEY)

col = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
       'AMT_CREDIT-dby-AMT_ANNUITY', 'DAYS_BIRTH']

train = utils.load_train([KEY]+col)
test = utils.load_test([KEY]+col)

train.columns = [KEY] + ['app_'+c for c in train.columns[1:]]
test.columns  = [KEY] + ['app_'+c for c in test.columns[1:]]

col_init = train.columns.tolist()

# =============================================================================
# feature
# =============================================================================
base['cnt_total'] = gr.size()

base['AMT_CREDIT_SUM_sum'] = gr['AMT_CREDIT_SUM'].sum()
base['AMT_CREDIT_SUM_DEBT_sum'] = gr['AMT_CREDIT_SUM_DEBT'].sum()

base['DAYS_CREDIT_min'] = gr['DAYS_CREDIT'].min()
base['DAYS_CREDIT_max'] = gr['DAYS_CREDIT'].max()


base['cnt_active'] = gr_act.size()
base['DAYS_CREDIT_act_min'] = gr_act['DAYS_CREDIT'].min()
base['DAYS_CREDIT_act_max'] = gr_act['DAYS_CREDIT'].max()


base['cnt_closed'] = gr_cls.size()
base['DAYS_CREDIT_cls_min'] = gr_cls['DAYS_CREDIT'].min()
base['DAYS_CREDIT_cls_max'] = gr_cls['DAYS_CREDIT'].max()



base.reset_index(inplace=True)
# =============================================================================
# merge
# =============================================================================

def mk_feature(df):
    
    df['DAYS_CREDIT_min-m-DAYS_BIRTH'] = df['DAYS_CREDIT_min'] - df['app_DAYS_BIRTH']
    df['DAYS_CREDIT_max-m-DAYS_BIRTH'] = df['DAYS_CREDIT_max'] - df['app_DAYS_BIRTH']
    
    df['DAYS_CREDIT_act_min-m-DAYS_BIRTH'] = df['DAYS_CREDIT_act_min'] - df['app_DAYS_BIRTH']
    df['DAYS_CREDIT_act_max-m-DAYS_BIRTH'] = df['DAYS_CREDIT_act_max'] - df['app_DAYS_BIRTH']
    
    df['DAYS_CREDIT_cls_min-m-DAYS_BIRTH'] = df['DAYS_CREDIT_cls_min'] - df['app_DAYS_BIRTH']
    df['DAYS_CREDIT_cls_max-m-DAYS_BIRTH'] = df['DAYS_CREDIT_cls_max'] - df['app_DAYS_BIRTH']
    
    df['AMT_CREDIT_SUM_DEBT_sum-p-app_AMT_CREDIT'] = df['AMT_CREDIT_SUM_DEBT_sum'] + df['app_AMT_CREDIT']
    df['AMT_CREDIT_SUM_DEBT_sum-p-app_AMT_CREDIT-dby-AMT_INCOME_TOTAL'] = df['AMT_CREDIT_SUM_DEBT_sum-p-app_AMT_CREDIT'] / df['app_AMT_INCOME_TOTAL']
    
    return


train2 = pd.merge(train, base, on=KEY, how='left')
mk_feature(train2)

test2 = pd.merge(test, base, on=KEY, how='left')
mk_feature(test2)

# =============================================================================
# output
# =============================================================================
train2.drop(col_init, axis=1, inplace=True)
test2.drop(col_init, axis=1, inplace=True)
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)
