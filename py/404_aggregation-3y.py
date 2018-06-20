#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:30:13 2018

@author: Kazuki

based on
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils
utils.start(__file__)
#==============================================================================
PREF = 'cre_404_'

KEY = 'SK_ID_CURR'

month_start = -12*3 # -96
month_end   = -12*2 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance')
cre = cre[cre['MONTHS_BALANCE'].between(month_start, month_end)].drop('SK_ID_PREV', axis=1)


#num_aggregations = {
#    # TODO: optimize stats
#    'NUM_INSTALMENT_VERSION': ['nunique'],
#    'DPD': ['max', 'mean', 'sum', 'nunique'],
#    'DBD': ['max', 'mean', 'sum', 'nunique'],
#    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
#    'AMT_PAYMENT':    ['min', 'max', 'mean', 'sum'],
#    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
#    'amt_ratio': ['min', 'mean', 'var'],
#    'amt_delta': ['min', 'mean', 'var'],
#    'days_weighted_delay': ['max', 'mean', 'sum'],
#}
#
#
#col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = cre
    
#    li = []
#    for c1 in df.columns:
#        for c2 in col_cat:
#            if c1.startswith(c2+'_'):
#                li.append(c1)
#                break
#    
#    cat_aggregations = {}
#    for cat in li:
#        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['CRE_COUNT'] = df.groupby('SK_ID_CURR').size()
    df_agg.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

aggregate()



#==============================================================================
utils.end(__file__)

