#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:24:12 2018

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
PREF = 'cre_402_'

KEY = 'SK_ID_CURR'

month_start = -12*1 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance')
cre = cre[cre['MONTHS_BALANCE'].between(month_start, month_end)].drop('SK_ID_PREV', axis=1)



num_aggregations = {
#    # TODO: optimize stats
'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
 'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
 'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
 'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
 'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
 'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
 'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
 'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
 'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
 'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
 'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
 'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
 'NAME_CONTRACT_STATUS': ['min', 'max', 'mean', 'sum'],
 'SK_DPD': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
 'app_AMT_INCOME_TOTAL': ['min', 'max', 'mean', 'sum'],
 'app_AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
 'app_AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
 'app_AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
 'AMT_BALANCE-dby-AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
 'AMT_BALANCE-dby-app_AMT_INCOME_TOTAL': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over0': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over5': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over10': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over15': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over20': ['min', 'max', 'mean', 'sum'],
 'SK_DPD_diff_over25': ['min', 'max', 'mean', 'sum'],
}
#
#
col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

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
