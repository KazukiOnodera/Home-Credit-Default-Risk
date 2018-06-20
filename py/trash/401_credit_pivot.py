#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:04:41 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

# setting
month_limit = 12 # max: 96

month_round = 1

PREF = 'cre_401'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
#cre = pd.read_csv('/Users/kazuki.onodera/Home-Credit-Default-Risk/py/sample_cre.csv')
cre = utils.read_pickles('../data/credit_card_balance')
cre.drop('SK_ID_PREV', axis=1, inplace=True)
cre = cre[cre['MONTHS_BALANCE']>=-month_limit]

cre['month_round'] = (cre['MONTHS_BALANCE'] / month_round).map(np.floor)
cre.drop('MONTHS_BALANCE', axis=1, inplace=True)

# groupby other credit cards
gr = cre.groupby(['SK_ID_CURR', 'month_round'])
cre_ = gr.size()
cre_.name = 'cc_size'
cre_ = pd.concat([cre_, gr.sum()], axis=1).reset_index() # TODO:NAME_CONTRACT_STATUS

col_app = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
           'AMT_CREDIT-dby-AMT_ANNUITY', 'DAYS_BIRTH']
cre_ = utils.merge(cre_, col_app)

cre_.sort_values(['SK_ID_CURR', 'month_round'], ascending=[True, False], inplace=True)


cre_['AMT_BALANCE-dby-AMT_CREDIT_LIMIT_ACTUAL']                   = cre_['AMT_BALANCE'] / cre_['AMT_CREDIT_LIMIT_ACTUAL']
cre_['AMT_DRAWINGS_ATM_CURRENT-dby-CNT_DRAWINGS_ATM_CURRENT']     = cre_['AMT_DRAWINGS_ATM_CURRENT'] / cre_['CNT_DRAWINGS_ATM_CURRENT']
cre_['AMT_DRAWINGS_CURRENT-dby-CNT_DRAWINGS_CURRENT']             = cre_['AMT_DRAWINGS_CURRENT'] / cre_['CNT_DRAWINGS_CURRENT']
cre_['AMT_DRAWINGS_OTHER_CURRENT-dby-CNT_DRAWINGS_OTHER_CURRENT'] = cre_['AMT_DRAWINGS_OTHER_CURRENT'] / cre_['CNT_DRAWINGS_OTHER_CURRENT']
cre_['AMT_DRAWINGS_POS_CURRENT-dby-CNT_DRAWINGS_POS_CURRENT']     = cre_['AMT_DRAWINGS_POS_CURRENT'] / cre_['CNT_DRAWINGS_POS_CURRENT']
cre_['AMT_BALANCE-dby-AMT_INCOME_TOTAL'] = cre_['AMT_BALANCE'] / cre_['AMT_INCOME_TOTAL']
cre_['AMT_BALANCE-dby-AMT_INCOME_TOTAL'] = cre_['AMT_BALANCE'] / cre_['AMT_CREDIT']
cre_['AMT_BALANCE-dby-AMT_INCOME_TOTAL'] = cre_['AMT_BALANCE'] / cre_['AMT_ANNUITY']
#cre_['-dby-'] = cre_[''] / cre_['']
#cre_['-dby-'] = cre_[''] / cre_['']
#cre_['-dby-'] = cre_[''] / cre_['']

# TODO: pct_change & diff & rolling mean
gr = cre_.groupby(['SK_ID_CURR'])
cre_['AMT_BALANCE_pctchng-1']          = gr['AMT_BALANCE'].pct_change(-1)
cre_['AMT_DRAWINGS_CURRENT_pctchng-1'] = gr['AMT_DRAWINGS_CURRENT'].pct_change(-1)
#cre_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)
#cre_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)
#cre_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)


cre_.drop(col_app[1:], axis=1, inplace=True)

# =============================================================================
# pivot
# =============================================================================
pt = pd.pivot_table(cre_, index='SK_ID_CURR', columns=['month_round'])

pt.columns = [f'{PREF}_{c[0]}_t{int(c[1])}' for c in pt.columns]

pt.reset_index(inplace=True)


# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])

test = utils.load_test([KEY])


train_ = pd.merge(train, pt, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(train_, '../feature/train')

test_ = pd.merge(test, pt, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(test_,  '../feature/test')


#==============================================================================
utils.end(__file__)


