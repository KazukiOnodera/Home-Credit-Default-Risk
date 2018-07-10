#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 07:26:50 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool#, cpu_count
#NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f306_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
#col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate(args):
    
    path, pref = args
    df = utils.read_pickles(path)
    df = df[df['DAYS_INSTALMENT'].between(day_start, day_end)]
    df_key = df[[KEY, 'SK_ID_PREV']].drop_duplicates()
    
    
    
    df_agg = df.groupby('SK_ID_PREV').agg({**utils_agg.ins_num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['INS_COUNT'] = df.groupby('SK_ID_PREV').size()
    df_agg = df_agg.add_prefix(pref).reset_index()
    
    utils.remove_feature(df_agg, var_limit=0, sample_size=19999)
    
    df_agg = pd.merge(df_agg, df_key, on='SK_ID_PREV', how='left').drop('SK_ID_PREV', axis=1)
    
    df_agg2 = df_agg.groupby(KEY).agg(['mean', 'var'])
    df_agg2.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg2.columns.tolist()])
    df_agg2.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg2, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg2, on=KEY, how='left').drop(KEY, axis=1)
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

pool = Pool(3)
pool.map(aggregate, argss)
pool.close()




#==============================================================================
utils.end(__file__)
