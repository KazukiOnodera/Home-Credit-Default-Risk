#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:17:36 2018

@author: kazuki.onodera

prev + bureau

"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f701_'

KEY = 'SK_ID_CURR'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
col = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
       'AMT_CREDIT-dby-AMT_ANNUITY', 'DAYS_BIRTH']

train = utils.load_train()
test = utils.load_test()

di = {'app_SK_ID_CURR': 'SK_ID_CURR'}
trte = pd.concat([train, test], ignore_index=True).add_prefix('app_').rename(columns=di)
base = trte[[KEY]].drop_duplicates().set_index(KEY)
trte_gr = trte.groupby('SK_ID_CURR')


# 'prev_AMT_ANNUITY', 'prev_AMT_APPLICATION', 'prev_AMT_CREDIT', 'prev_AMT_DOWN_PAYMENT', 'prev_AMT_GOODS_PRICE'
# 'prev_total_debt', 'prev_amt_paid'
di = {'prev_SK_ID_CURR': 'SK_ID_CURR', 'prev_SK_ID_PREV': 'SK_ID_PREV'}
prev = utils.read_pickles('../data/previous_application').add_prefix('prev_').rename(columns=di)
prev_gr     = prev.groupby(KEY)
prev_gr_app = prev[prev['prev_NAME_CONTRACT_STATUS']=='Approved'].groupby(KEY)
prev_gr_ref = prev[prev['prev_NAME_CONTRACT_STATUS']=='Refused'].groupby(KEY)
prev_gr_act = prev[prev['prev_active']==1].groupby(KEY)
prev_gr_cmp = prev[prev['prev_completed']==1].groupby(KEY)


# 'bure_CREDIT_CURRENCY', 'bure_DAYS_CREDIT', 'bure_CREDIT_DAY_OVERDUE',
#   'bure_DAYS_CREDIT_ENDDATE', 'bure_DAYS_ENDDATE_FACT',
#   'bure_AMT_CREDIT_MAX_OVERDUE', 'bure_CNT_CREDIT_PROLONG',
#   'bure_AMT_CREDIT_SUM', 'bure_AMT_CREDIT_SUM_DEBT',
#   'bure_AMT_CREDIT_SUM_LIMIT', 'bure_AMT_CREDIT_SUM_OVERDUE'
di = {'bure_SK_ID_CURR': 'SK_ID_CURR', 'bure_SK_ID_BUREAU': 'SK_ID_BUREAU'}
bure = utils.read_pickles('../data/bureau').add_prefix('bure_').rename(columns=di)
bure_gr     = bure.groupby(KEY)
bure_gr_act = bure[bure['bure_CREDIT_ACTIVE']=='Active'].groupby(KEY)
bure_gr_cls = bure[bure['bure_CREDIT_ACTIVE']=='Closed'].groupby(KEY)



# =============================================================================
# 
# =============================================================================

#### all_credit ####
all_credit = pd.concat([trte_gr['app_AMT_CREDIT'].sum(), 
                        prev_gr_app['prev_AMT_CREDIT'].sum(), 
                        bure_gr['bure_AMT_CREDIT_SUM'].sum()], axis=1)

base['all_credit_min']  = all_credit.min(1)
base['all_credit_max']  = all_credit.max(1)
base['all_credit_sum']  = all_credit.sum(1)
base['all_credit_mean'] = all_credit.mean(1)
base['all_credit_std']  = all_credit.std(1)

base['all_credit_min-dby-income']  = base['all_credit_min']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit_max-dby-income']  = base['all_credit_max']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit_sum-dby-income']  = base['all_credit_sum']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit_mean-dby-income'] = base['all_credit_mean'] / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit_std-dby-income']  = base['all_credit_std']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()



#### all_credit-prevact ####
all_credit = pd.concat([trte_gr['app_AMT_CREDIT'].sum(), 
                        prev_gr_act['prev_AMT_CREDIT'].sum(), 
                        bure_gr['bure_AMT_CREDIT_SUM'].sum()], axis=1)

base['all_credit-prevact_min']  = all_credit.min(1)
base['all_credit-prevact_max']  = all_credit.max(1)
base['all_credit-prevact_sum']  = all_credit.sum(1)
base['all_credit-prevact_mean'] = all_credit.mean(1)
base['all_credit-prevact_std']  = all_credit.std(1)

base['all_credit-prevact_min-dby-income']  = base['all_credit-prevact_min']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit-prevact_max-dby-income']  = base['all_credit-prevact_max']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit-prevact_sum-dby-income']  = base['all_credit-prevact_sum']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit-prevact_mean-dby-income'] = base['all_credit-prevact_mean'] / trte_gr['app_AMT_INCOME_TOTAL'].sum()
base['all_credit-prevact_std-dby-income']  = base['all_credit-prevact_std']  / trte_gr['app_AMT_INCOME_TOTAL'].sum()







# =============================================================================
# output
# =============================================================================
base.reset_index(inplace=True)
train = pd.merge(utils.load_train([KEY]), base, on=KEY, how='left').drop(KEY, axis=1)
test = pd.merge(utils.load_test([KEY]), base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')



#==============================================================================
utils.end(__file__)


