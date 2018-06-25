#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:39:07 2018

@author: kazuki.onodera

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
PREF = 'bure_501_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # -2922
day_end   = -365*0 # -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
bure = utils.read_pickles('../data/bureau')
bure = bure[bure['DAYS_CREDIT'].between(day_start, day_end)]


stats = ['min', 'max', 'mean', 'sum', 'var']

num_aggregations = {
        # TODO: optimize stats
        'DAYS_CREDIT': stats,
        'CREDIT_DAY_OVERDUE': stats,
        'DAYS_CREDIT_ENDDATE': stats,
        'DAYS_ENDDATE_FACT': stats,
        'AMT_CREDIT_MAX_OVERDUE': stats,
        'CNT_CREDIT_PROLONG': stats,
        'AMT_CREDIT_SUM': stats,
        'AMT_CREDIT_SUM_DEBT': stats,
        'AMT_CREDIT_SUM_LIMIT': stats,
        'AMT_CREDIT_SUM_OVERDUE': stats,
        'CREDIT_TYPE': stats,
        'DAYS_CREDIT_UPDATE': stats,
        'AMT_ANNUITY': stats,
        
        'credit-dby-income': stats,
        'AMT_CREDIT_SUM_DEBT-dby-income': stats,
        'AMT_CREDIT_SUM_LIMIT-dby-income': stats,
        'AMT_CREDIT_SUM_OVERDUE-dby-income': stats,
        'credit-dby-annuity': stats,
        'AMT_CREDIT_SUM_DEBT-dby-annuity': stats,
        'AMT_CREDIT_SUM_LIMIT-dby-annuity': stats,
        'AMT_CREDIT_SUM_OVERDUE-dby-annuity': stats,
        'DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT': stats,
        'DAYS_ENDDATE_FACT-m-DAYS_CREDIT': stats,
        'DAYS_ENDDATE_FACT-m-DAYS_CREDIT_ENDDATE': stats,
        'DAYS_CREDIT_UPDATE-m-DAYS_CREDIT': stats,
        'DAYS_CREDIT_UPDATE-m-DAYS_CREDIT_ENDDATE': stats,
        'DAYS_CREDIT_UPDATE-m-DAYS_ENDDATE_FACT': stats,
        'AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT': stats,
        'AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM': stats,
        'AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_LIMIT': stats,
        'AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_LIMIT': stats,
        'AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT': stats,
        'AMT_CREDIT_SUM-dby-debt-p-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT': stats,
}


col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = utils.get_dummies(bure)
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['BURE_COUNT'] = df.groupby('SK_ID_CURR').size()
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
