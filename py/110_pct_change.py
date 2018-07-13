#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:32:11 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f110_'

KEY = 'SK_ID_CURR'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

col_num = [ 
'AMT_ANNUITY',
 'AMT_APPLICATION',
 'AMT_CREDIT',
 'AMT_DOWN_PAYMENT',
 'AMT_GOODS_PRICE',
 'RATE_DOWN_PAYMENT',
 'RATE_INTEREST_PRIMARY',
 'RATE_INTEREST_PRIVILEGED',
 'DAYS_DECISION',
 'SELLERPLACE_AREA',
 'CNT_PAYMENT',
 'DAYS_FIRST_DRAWING',
 'DAYS_FIRST_DUE',
 'DAYS_LAST_DUE_1ST_VERSION',
 'DAYS_LAST_DUE',
 'DAYS_TERMINATION',
 'NFLAG_INSURED_ON_APPROVAL',
 'days_fdue-m-fdrw',
 'days_ldue1-m-fdrw',
 'days_ldue-m-fdrw',
 'days_trm-m-fdrw',
 'days_ldue1-m-fdue',
 'days_ldue-m-fdue',
 'days_trm-m-fdue',
 'days_ldue-m-ldue1',
 'days_trm-m-ldue1',
 'days_trm-m-ldue',
 'total_debt',
 'AMT_CREDIT-d-total_debt',
 'AMT_GOODS_PRICE-d-total_debt',
 'AMT_GOODS_PRICE-d-AMT_CREDIT',
 'AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
 'AMT_APPLICATION-d-app_AMT_INCOME_TOTAL',
 'AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
 'AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
 'AMT_ANNUITY-m-app_AMT_INCOME_TOTAL',
 'AMT_APPLICATION-m-app_AMT_INCOME_TOTAL',
 'AMT_CREDIT-m-app_AMT_INCOME_TOTAL',
 'AMT_GOODS_PRICE-m-app_AMT_INCOME_TOTAL',
 'AMT_ANNUITY-d-app_AMT_CREDIT',
 'AMT_APPLICATION-d-app_AMT_CREDIT',
 'AMT_CREDIT-d-app_AMT_CREDIT',
 'AMT_GOODS_PRICE-d-app_AMT_CREDIT',
 'AMT_ANNUITY-m-app_AMT_CREDIT',
 'AMT_APPLICATION-m-app_AMT_CREDIT',
 'AMT_CREDIT-m-app_AMT_CREDIT',
 'AMT_GOODS_PRICE-m-app_AMT_CREDIT',
 'AMT_ANNUITY-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
 'AMT_APPLICATION-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
 'AMT_CREDIT-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
 'AMT_GOODS_PRICE-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
 'AMT_ANNUITY-d-app_AMT_ANNUITY',
 'AMT_APPLICATION-d-app_AMT_ANNUITY',
 'AMT_CREDIT-d-app_AMT_ANNUITY',
 'AMT_GOODS_PRICE-d-app_AMT_ANNUITY',
 'AMT_ANNUITY-m-app_AMT_ANNUITY',
 'AMT_APPLICATION-m-app_AMT_ANNUITY',
 'AMT_CREDIT-m-app_AMT_ANNUITY',
 'AMT_GOODS_PRICE-m-app_AMT_ANNUITY',
 'AMT_ANNUITY-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
 'AMT_APPLICATION-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
 'AMT_CREDIT-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
 'AMT_GOODS_PRICE-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
 'AMT_ANNUITY-d-app_AMT_GOODS_PRICE',
 'AMT_APPLICATION-d-app_AMT_GOODS_PRICE',
 'AMT_CREDIT-d-app_AMT_GOODS_PRICE',
 'AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE',
 'AMT_ANNUITY-m-app_AMT_GOODS_PRICE',
 'AMT_APPLICATION-m-app_AMT_GOODS_PRICE',
 'AMT_CREDIT-m-app_AMT_GOODS_PRICE',
 'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE',
 'AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
 'AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
 'AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
 'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
]


train = utils.load_train([KEY])
test = utils.load_test([KEY])

prev = utils.read_pickles('../data/previous_application', ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']+col_num)
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True)

base = prev[[KEY]]


gr1 = prev.groupby(KEY)
gr2 = prev[prev['NAME_CONTRACT_TYPE']=='Consumer loans'].groupby(KEY)
gr3 = prev[prev['NAME_CONTRACT_TYPE']=='Cash loans'].groupby(KEY)
gr4 = prev[prev['NAME_CONTRACT_TYPE']=='Revolving loans'].groupby(KEY)
gr5 = prev[prev['NAME_CONTRACT_TYPE'].isnull()].groupby(KEY)



# =============================================================================
# 
# =============================================================================

def multi(feature_name):
    # gr1
    pref = ''
    c = pref + feature_name + '_pct_change'
    base[c] = gr1[feature_name].pct_change()
    
    gr = base.groupby(KEY)
    
    feature = pd.concat([gr[c].min(), gr[c].mean(), gr[c].max(), 
                         gr[c].median(), gr[c].var()], axis=1)
    
    feature.columns = [c+'_min', c+'_mean', c+'_max', c+'_median', c+'_var']
    
    utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    feature.reset_index(inplace=True)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    
    # gr2
    pref = 'con_'
    c = pref + feature_name + '_pct_change'
    base[c] = gr2[feature_name].pct_change()
    
    gr = base.groupby(KEY)
    
    feature = pd.concat([gr[c].min(), gr[c].mean(), gr[c].max(), 
                         gr[c].median(), gr[c].var()], axis=1)
    
    feature.columns = [c+'_min', c+'_mean', c+'_max', c+'_median', c+'_var']
    
    utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    feature.reset_index(inplace=True)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    
    # gr3
    pref = 'csh_'
    c = pref + feature_name + '_pct_change'
    base[c] = gr3[feature_name].pct_change()
    
    gr = base.groupby(KEY)
    
    feature = pd.concat([gr[c].min(), gr[c].mean(), gr[c].max(), 
                         gr[c].median(), gr[c].var()], axis=1)
    
    feature.columns = [c+'_min', c+'_mean', c+'_max', c+'_median', c+'_var']
    
    utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    feature.reset_index(inplace=True)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    
    # gr4
    pref = 'rev_'
    c = pref + feature_name + '_pct_change'
    base[c] = gr4[feature_name].pct_change()
    
    gr = base.groupby(KEY)
    
    feature = pd.concat([gr[c].min(), gr[c].mean(), gr[c].max(), 
                         gr[c].median(), gr[c].var()], axis=1)
    
    feature.columns = [c+'_min', c+'_mean', c+'_max', c+'_median', c+'_var']
    
    utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    feature.reset_index(inplace=True)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    # gr5
    pref = 'rev_'
    c = pref + feature_name + '_pct_change'
    base[c] = gr5[feature_name].pct_change()
    
    gr = base.groupby(KEY)
    
    feature = pd.concat([gr[c].min(), gr[c].mean(), gr[c].max(), 
                         gr[c].median(), gr[c].var()], axis=1)
    
    feature.columns = [c+'_min', c+'_mean', c+'_max', c+'_median', c+'_var']
    
    utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    feature.reset_index(inplace=True)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return

# =============================================================================
# main
# =============================================================================


pool = Pool(len(col_num))
callback = pool.map(multi, col_num)
pool.close()







#==============================================================================
utils.end(__file__)
