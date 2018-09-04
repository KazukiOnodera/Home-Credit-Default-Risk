#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 00:23:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f155_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/future_application')
base = prev[[KEY]].drop_duplicates().set_index(KEY)

gr = prev.groupby(KEY)
gr_app = prev[prev['NAME_CONTRACT_STATUS']=='Approved'].groupby(KEY)
gr_ref = prev[prev['NAME_CONTRACT_STATUS']=='Refused'].groupby(KEY)
gr_act = prev[prev['active']==1].groupby(KEY)
gr_cmp = prev[prev['completed']==1].groupby(KEY)

col = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
       'AMT_CREDIT-d-AMT_ANNUITY', 'DAYS_BIRTH']
train = utils.load_train([KEY]+col)
test = utils.load_test([KEY]+col)

train.AMT_ANNUITY.fillna(0, inplace=True)
test.AMT_ANNUITY.fillna(0, inplace=True)


train.columns = [KEY] + ['app_'+c for c in train.columns[1:]]
test.columns  = [KEY] + ['app_'+c for c in test.columns[1:]]

col_init = train.columns.tolist()

# =============================================================================
# feature
# =============================================================================

# size
base['approved_cnt']  = gr_app.size()
base['resfused_cnt']  = gr_ref.size()
base['active_cnt']  = gr_act.size()
base['completed_cnt']  = gr_cmp.size()

base['app-p-ref_cnt'] = base['approved_cnt'] + base['resfused_cnt']
base['approved_ratio'] = base['approved_cnt'] / base['app-p-ref_cnt']
base['active_ratio']   = base['active_cnt'] / base['approved_cnt']


c = 'total_debt'
base[f'{c}_min']  = gr_app[c].min()
base[f'{c}_mean'] = gr_app[c].mean()
base[f'{c}_max']  = gr_app[c].max()
base[f'{c}_sum'] = gr_app[c].sum()

c = 'amt_paid'
base[f'{c}_min']  = gr_app[c].min()
base[f'{c}_mean'] = gr_app[c].mean()
base[f'{c}_max']  = gr_app[c].max()
base[f'{c}_sum'] = gr_app[c].sum()

c = 'amt_unpaid'
base[f'{c}_min']  = gr_app[c].min()
base[f'{c}_mean'] = gr_app[c].mean()
base[f'{c}_max']  = gr_app[c].max()
base[f'{c}_sum'] = gr_app[c].sum()

c = 'cnt_paid'
base[f'{c}_min']  = gr_app[c].min()
base[f'{c}_mean'] = gr_app[c].mean()
base[f'{c}_max']  = gr_app[c].max()
base[f'{c}_sum'] = gr_app[c].sum()

c = 'cnt_unpaid'
base[f'{c}_min']  = gr_app[c].min()
base[f'{c}_mean'] = gr_app[c].mean()
base[f'{c}_max']  = gr_app[c].max()
base[f'{c}_sum'] = gr_app[c].sum()

# ratio
base['amt_paid-d-unpaid_min']  = base['amt_paid_min']  / base['amt_unpaid_min']
base['amt_paid-d-unpaid_mean'] = base['amt_paid_mean'] / base['amt_unpaid_mean']
base['amt_paid-d-unpaid_max']  = base['amt_paid_max']  / base['amt_unpaid_max']
base['amt_paid-d-unpaid_sum']  = base['amt_paid_sum']  / base['amt_unpaid_sum']

base['cnt_paid-d-unpaid_min']  = base['cnt_paid_min']  / base['cnt_unpaid_min']
base['cnt_paid-d-unpaid_mean'] = base['cnt_paid_mean'] / base['cnt_unpaid_mean']
base['cnt_paid-d-unpaid_max']  = base['cnt_paid_max']  / base['cnt_unpaid_max']
base['cnt_paid-d-unpaid_sum']  = base['cnt_paid_sum']  / base['cnt_unpaid_sum']





c = 'AMT_ANNUITY'
base[f'{c}_act_min']  = gr_act[c].min()
base[f'{c}_act_mean'] = gr_act[c].mean()
base[f'{c}_act_max']  = gr_act[c].max()
base[f'{c}_act_sum'] = gr_act[c].sum()

base[f'{c}_cmp_min']  = gr_cmp[c].min()
base[f'{c}_cmp_mean'] = gr_cmp[c].mean()
base[f'{c}_cmp_max']  = gr_cmp[c].max()
base[f'{c}_cmp_sum'] = gr_cmp[c].sum()

c = 'DAYS_LAST_DUE_1ST_VERSION'
base[f'{c}_act_min']  = gr_act[c].min()
base[f'{c}_act_mean'] = gr_act[c].mean()
base[f'{c}_act_max']  = gr_act[c].max()
base[f'{c}_act_sum'] = gr_act[c].sum()


#base['active_cnt'] = gr_app['active'].sum()
#base['active_ratio'] = gr_app['active'].mean()
#base['completed_cnt'] = gr_app['completed'].sum()

base['DAYS_DECISION_min'] = gr['DAYS_DECISION'].min()
base['DAYS_DECISION_max'] = gr['DAYS_DECISION'].max()

base['amt_paid_sum-d-total_debt_sum'] = base['amt_paid_sum'] / base['total_debt_sum']
base['amt_paid_sum-d-amt_unpaid_sum'] = base['amt_paid_sum'] / base['amt_unpaid_sum']


# app, ref
#base['cnt_approved'] = gr_app.size()
#base['cnt_refused'] = gr_ref.size()
#base['approved_ratio'] = base['cnt_approved'] / base['cnt_approved'] + base['cnt_refused'] 

base['DAYS_DECISION_app_min'] = gr_app['DAYS_DECISION'].min()
base['DAYS_DECISION_app_max'] = gr_app['DAYS_DECISION'].max()

base['DAYS_DECISION_ref_min'] = gr_ref['DAYS_DECISION'].min()
base['DAYS_DECISION_ref_max'] = gr_ref['DAYS_DECISION'].max()



# future payment
col = prev.head().filter(regex='^future_payment_').columns
col_future_sum  = []
col_future_min  = []
col_future_mean = []
col_future_max  = []
for c in col:
    base[f'{c}_sum'] = gr_act[c].sum()
    col_future_sum.append(f'{c}_sum')
    base[f'{c}_min'] = gr_act[c].min()
    col_future_min.append(f'{c}_min')
    base[f'{c}_max'] = gr_act[c].max()
    col_future_max.append(f'{c}_max')
    base[f'{c}_mean'] = gr_act[c].mean()
    col_future_mean.append(f'{c}_mean')

# past payment
col = prev.head().filter(regex='^past_payment_').columns
col_past_sum = []
col_past_min = []
col_past_mean = []
col_past_max = []
for c in col:
    base[f'{c}_sum'] = gr_app[c].sum()
    col_past_sum.append(f'{c}_sum')
    base[f'{c}_min'] = gr_app[c].min()
    col_past_min.append(f'{c}_min')
    base[f'{c}_mean'] = gr_app[c].mean()
    col_past_mean.append(f'{c}_mean')
    base[f'{c}_max'] = gr_app[c].max()
    col_past_max.append(f'{c}_max')


base.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

def mk_feature(df):
    
    df['total_debt_sum-p-app']            = df['total_debt_sum'] + df['app_AMT_CREDIT']
    df['total_debt_sum-p-app-d-income'] = df['total_debt_sum-p-app'] / df['app_AMT_INCOME_TOTAL']
    df['amt_unpaid_sum-p-app']            = df['amt_unpaid_sum'] + df['app_AMT_CREDIT']
    df['amt_unpaid_sum-p-app-d-income'] = df['amt_unpaid_sum-p-app'] / df['app_AMT_INCOME_TOTAL']

    # future payment
    df[col_future_sum+col_past_sum] = df[col_future_sum+col_past_sum].fillna(0)
    col_1 = []
    df['tmp'] = df['app_AMT_CREDIT-d-AMT_ANNUITY'].map(np.ceil)
    for i,c in enumerate( col_future_sum ):
        c1 = f'prevapp_future_payment_{i+1}m'
        df[c1] = df[c] + df['tmp'].map(lambda x: min(x, 1)) * df['app_AMT_ANNUITY']
        df['tmp'] -= 1
        df['tmp'] = df['tmp'].map(lambda x: max(x, 0))
        col_1.append(c1)
        
    del df['tmp']
    
    df['prevapp_future_payment_1vs2'] = df['prevapp_future_payment_1m'] / df['prevapp_future_payment_2m']
    df['prevapp_future_payment_1vs3'] = df['prevapp_future_payment_1m'] / df['prevapp_future_payment_3m']
    df['prevapp_future_payment_1vs4'] = df['prevapp_future_payment_1m'] / df['prevapp_future_payment_4m']
    df['prevapp_future_payment_1vs5'] = df['prevapp_future_payment_1m'] / df['prevapp_future_payment_5m']
    df['prevapp_future_payment_1vs6'] = df['prevapp_future_payment_1m'] / df['prevapp_future_payment_6m']
    
    # future
    df['prevapp_future_payment_max'] = df[col_1].max(1) # next month total
    df['prev_future_payment_max'] = df['prevapp_future_payment_max'] - df['app_AMT_ANNUITY'] # without app
    df['future_payment_app_ratio'] = df['prevapp_future_payment_max'] / df['prev_future_payment_max']
    df['prevapp_future_payment_max-d-income']  = df['prevapp_future_payment_max'] / df['app_AMT_INCOME_TOTAL']
    df['prevapp_future_payment_max-d-credit']  = df['prevapp_future_payment_max'] / df['app_AMT_CREDIT']
    df['prevapp_future_payment_max-d-annuity'] = df['prevapp_future_payment_max'] / df['app_AMT_ANNUITY']
    df['prev_future_payment_max-d-income']     = df['prev_future_payment_max']    / df['app_AMT_INCOME_TOTAL']
    df['prev_future_payment_max-d-credit']     = df['prev_future_payment_max']    / df['app_AMT_CREDIT']
    df['prev_future_payment_max-d-annuity']    = df['prev_future_payment_max']    / df['app_AMT_ANNUITY']
    
    # past
    df['past_payment_sum_max'] = df[col_past_sum].max(1) # past max
    df['past_payment_sum_max-d-income'] = df['past_payment_sum_max'] / df['app_AMT_INCOME_TOTAL']
    df['past_payment_sum_max-d-credit'] = df['past_payment_sum_max'] / df['app_AMT_CREDIT']
    df['past_payment_sum_max-d-annuity'] = df['past_payment_sum_max'] / df['app_AMT_ANNUITY']
    
    # future vs past
    df['future_vs_past_max'] = df['prevapp_future_payment_max'] / df['past_payment_sum_max']
    df['future_vs_past_max_withoutapp'] = df['prev_future_payment_max'] / df['past_payment_sum_max']
    df['future_vs_past_max-vs-withoutapp'] = df['future_vs_past_max'] / df['future_vs_past_max_withoutapp']
    df['future_vs_past_max-d-income'] = df['prevapp_future_payment_max-d-income'] / df['past_payment_sum_max-d-income']
    
    df['DAYS_DECISION_min-m-DAYS_BIRTH'] = df['DAYS_DECISION_min'] - df['app_DAYS_BIRTH']
    df['DAYS_DECISION_max-m-DAYS_BIRTH'] = df['DAYS_DECISION_max'] - df['app_DAYS_BIRTH']



train2 = pd.merge(train, base, on=KEY, how='left')
mk_feature(train2)


test2 = pd.merge(test, base, on=KEY, how='left')
mk_feature(test2)

#utils.remove_feature(train2)
#train2, test2 = train2.align(test2, join='inner', axis=1)

# =============================================================================
# output
# =============================================================================
train2.drop(col_init, axis=1, inplace=True)
test2.drop(col_init, axis=1, inplace=True)
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)


