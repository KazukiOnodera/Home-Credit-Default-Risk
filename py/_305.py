#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:00:34 2018

@author: kazuki.onodera
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

# =============================================================================
# feature
# =============================================================================

ins['months_installment']   = (ins['DAYS_INSTALMENT'].fillna(1)/30).map(np.floor)
ins['months_entry_payment'] = (ins['DAYS_ENTRY_PAYMENT'].fillna(1)/30).map(np.floor)

gr = ins.drop_duplicates(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).groupby(['SK_ID_CURR', 'months_installment'])
ins_month = gr.size()
ins_month.name = f'{PREF}_inst_size'
ins_month = ins_month.to_frame()
ins_month[f'{PREF}_AMT_INSTALMENT_sum'] = gr['AMT_INSTALMENT'].sum()

gr = ins.groupby(['SK_ID_CURR', 'months_entry_payment'])
tmp = gr.size()
tmp.name = f'{PREF}_pay_size'
ins_month_ = pd.concat([ins_month, tmp], axis=1)
#ins_month[f'{PREF}_pay_size'] = gr.size()
ins_month_[f'{PREF}_AMT_PAYMENT_sum'] = gr['AMT_PAYMENT'].sum()






ins['NUM_INSTALMENT_VERSION_diff'] = ins.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].diff(-1)
ins['NUM_INSTALMENT_VERSION_dec']  = (ins['NUM_INSTALMENT_VERSION_diff'] < 0)*1
ins['NUM_INSTALMENT_VERSION_inc']  = (ins['NUM_INSTALMENT_VERSION_diff'] > 0)*1

#ins['NUM_INSTALMENT_NUMBER_diff'] = ins.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].diff(-1)
ins['days_delayed_payment_dec-th0']  = (ins['days_delayed_payment'] < 0)*1
ins['days_delayed_payment_inc-th0']  = (ins['days_delayed_payment'] > 0)*1

def to_decimal(x):
    x = ''.join(map(str, x))[::-1]
    return float(x[0] + '.' + x[1:])








# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(train, f'../data/{No}_train', utils.SPLIT_SIZE)
del train; gc.collect()

test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(test,  f'../data/{No}_test',  utils.SPLIT_SIZE)






#==============================================================================
utils.end(__file__)


