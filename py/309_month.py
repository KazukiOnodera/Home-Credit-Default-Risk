#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:07:36 2018

@author: kazuki.onodera

month

"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f309_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

train = utils.load_train([KEY])
test = utils.load_test([KEY])

COL = ['AMT_PAYMENT', 'AMT_PAYMENT-d-app_AMT_INCOME_TOTAL', 'AMT_PAYMENT-d-app_AMT_CREDIT',
       'AMT_PAYMENT-d-app_AMT_ANNUITY', 'AMT_PAYMENT-d-app_AMT_GOODS_PRICE']

num_agg = {}
for c in COL:
    num_agg[c] = ['min', 'mean', 'max', 'var']

# =============================================================================
# 
# =============================================================================
def aggregate(args):
    path, pref = args
    
    df = utils.read_pickles(path, [KEY, 'SK_ID_PREV', 'month']+COL)
#    df = df[df['DAYS_INSTALMENT'].between(day_start, day_end)]
#    del df['SK_ID_PREV']
    
    df = df.groupby([KEY, 'SK_ID_PREV', 'month'])[COL].sum().reset_index()
    
    df_agg = df.groupby(KEY).agg({**num_agg})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['INS_COUNT'] = df.groupby(KEY).size()
    df_agg = df_agg.add_prefix(pref).reset_index()
    
    utils.remove_feature(df_agg, var_limit=0, sample_size=19999)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================
argss = [
        ['../data/installments_payments', ''],
        ['../data/installments_payments_delay', 'delay_'],
        ['../data/installments_payments_notdelay', 'notdelay_'],
        ]
pool = Pool(len(argss))
pool.map(aggregate, argss)
pool.close()




#==============================================================================
utils.end(__file__)
