#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:59:32 2018

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
PREF = 'bureau_502'
NTHREAD = 3


col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

col_group = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

# =============================================================================
# feature
# =============================================================================
bureau = utils.read_pickles('../data/bureau')

bureau = bureau[bureau['DAYS_CREDIT'].between(-365, 0)]
bureau = utils.get_dummies(bureau)
bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
gr = bureau.groupby(KEY)

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


