#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:27:48 2018

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
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f403_'

KEY = 'SK_ID_PREV'

month_start = -12*2 # -96
month_end   = -12*1 # -96

os.system(f'rm ../feature_prev/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance')
cre = cre[cre['MONTHS_BALANCE'].between(month_start, month_end)]


col_cat = ['NAME_CONTRACT_STATUS']

train = utils.read_pickles('../data/prev_train', [KEY])
test  = utils.read_pickles('../data/prev_test', [KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = utils.get_dummies(cre)
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby(KEY).agg({**utils_agg.cre_num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['CRE_COUNT'] = df.groupby(KEY).size()
    df_agg.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature_prev/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature_prev/test')
    
    return

# =============================================================================
# main
# =============================================================================

aggregate()



#==============================================================================
utils.end(__file__)

