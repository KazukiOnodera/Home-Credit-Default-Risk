#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 03:18:02 2018

@author: Kazuki
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

PREF = 'pos_201'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
#pos = pd.read_csv('/Users/Kazuki/Home-Credit-Default-Risk/py/sample_POS.csv')
pos = utils.read_pickles('../data/POS_CASH_balance')
pos.drop('SK_ID_PREV', axis=1, inplace=True)
pos = pos[pos['MONTHS_BALANCE']>=-month_limit]

pos['month_round'] = (pos['MONTHS_BALANCE'] / month_round).map(np.floor)
pos.drop('MONTHS_BALANCE', axis=1, inplace=True)

# groupby other credit cards
gr = pos.groupby(['SK_ID_CURR', 'month_round'])
pos_ = gr.size()
pos_.name = 'pos_size'
pos_ = pd.concat([pos_, gr.sum()], axis=1).reset_index() # TODO:NAME_CONTRACT_STATUS
pos_.sort_values(['SK_ID_CURR', 'month_round'], ascending=[True, False], inplace=True)

pos_['CNT_INSTALMENT_FUTURE-dby-CNT_INSTALMENT']   = pos_['CNT_INSTALMENT_FUTURE'] / pos_['CNT_INSTALMENT']
#pos_['-by-'] = pos_[''] / pos_['']
#pos_['-by-'] = pos_[''] / pos_['']
#pos_['-by-'] = pos_[''] / pos_['']
#pos_['-by-'] = pos_[''] / pos_['']

# TODO: pct_change & diff & rolling mean
#gr = pos_.groupby(['SK_ID_CURR'])
#pos_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)
#pos_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)
#pos_['AMT_BALANCE_pctchng-1'] = gr['AMT_BALANCE'].pct_change(-1)

pt = pd.pivot_table(pos_, index='SK_ID_CURR', columns=['month_round'])

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



