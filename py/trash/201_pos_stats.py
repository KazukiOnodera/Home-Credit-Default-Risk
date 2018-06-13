#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:03 2018

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
PREF = 'pos_201'

NTHREAD = 2

col_num = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
           'SK_DPD', 'SK_DPD_DEF']

col_cat = ['NAME_CONTRACT_STATUS']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
pos = utils.get_dummies(utils.read_pickles('../data/POS_CASH_balance'))
pos.drop('SK_ID_PREV', axis=1, inplace=True)

base = pos[[KEY]].drop_duplicates().set_index(KEY)

gr = pos.groupby(KEY)

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


