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


gr = cre.groupby(KEY)

train = utils.load_train([KEY])

test = utils.load_test([KEY])

def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================

def multi(p):
    
    if p==0:
        feature = gr.size()
        feature.name = f'{PREF}_{KEY}_size'
        feature = feature.reset_index()
    elif p==1:
        feature = gr.min().add_prefix(f'{PREF}_').add_suffix('_min').reset_index()
    elif p==2:
        feature = gr.max().add_prefix(f'{PREF}_').add_suffix('_max').reset_index()
    elif p==3:
        feature = gr.mean().add_prefix(f'{PREF}_').add_suffix('_mean').reset_index()
    elif p==4:
        feature = gr.std().add_prefix(f'{PREF}_').add_suffix('_std').reset_index()
    elif p==5:
        feature = gr.sum().add_prefix(f'{PREF}_').add_suffix('_sum').reset_index()
    elif p==6:
        feature = gr.quantile(0.25).add_prefix(f'{PREF}_').add_suffix('_q25').reset_index()
    elif p==7:
        feature = gr.quantile(0.50).add_prefix(f'{PREF}_').add_suffix('_q50').reset_index()
    elif p==8:
        feature = gr.quantile(0.75).add_prefix(f'{PREF}_').add_suffix('_q75').reset_index()
    else:
        return
    
    train_ = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(train_, '../feature/train')
    
    test_ = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(test_,  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================
pool = Pool(10)
pool.map(multi, range(10))
pool.close()


#==============================================================================
utils.end(__file__)


