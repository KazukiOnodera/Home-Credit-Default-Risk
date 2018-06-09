#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:35:23 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'cre_401'

#col_num = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
#           'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
#           'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
#           'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
#           'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
#           'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
#           'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
#           'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF']
#
#col_cat = ['CNT_DRAWINGS_OTHER_CURRENT', 'NAME_CONTRACT_STATUS']
#
#col_group = ['SK_ID_PREV', 'CNT_DRAWINGS_OTHER_CURRENT', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
cre = utils.get_dummies(utils.read_pickles('../data/credit_card_balance'))
cre.drop('SK_ID_PREV', axis=1, inplace=True)

base = cre[[KEY]].drop_duplicates().set_index(KEY)


def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================
gr = cre.groupby(KEY)

# stats
keyname = 'gby-'+KEY
base[f'{PREF}_{keyname}_size'] = gr.size()

base = pd.concat([
                base,
                gr.min().add_prefix(f'{PREF}_').add_suffix('_min'),
                gr.max().add_prefix(f'{PREF}_').add_suffix('_max'),
                gr.mean().add_prefix(f'{PREF}_').add_suffix('_mean'),
                gr.std().add_prefix(f'{PREF}_').add_suffix('_std'),
                gr.sum().add_prefix(f'{PREF}_').add_suffix('_sum'),
                gr.quantile(0.25).add_prefix(f'{PREF}_').add_suffix('_q25'),
                gr.quantile(0.50).add_prefix(f'{PREF}_').add_suffix('_q50'),
                gr.quantile(0.75).add_prefix(f'{PREF}_').add_suffix('_q75'),
                ], axis=1)


# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickle_each_cols(train, '../feature/train')
del train; gc.collect()


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickle_each_cols(test,  '../feature/test')



#==============================================================================
utils.end(__file__)


