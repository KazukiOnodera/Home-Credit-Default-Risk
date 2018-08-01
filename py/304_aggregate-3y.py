#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 00:47:12 2018

@author: Kazuki

based on
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
#NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f304_'

KEY = 'SK_ID_CURR'

day_start = -365*3  # min: -2922
day_end   = -365*2  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate(args):
    path, pref = args
    
    df = utils.read_pickles(path)
    df = df[df['DAYS_INSTALMENT'].between(day_start, day_end)]
    del df['SK_ID_PREV']
    
    df_agg = df.groupby(KEY).agg({**utils_agg.ins_num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        try:
            df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
            df_agg[f'{c}-m-min'] = df_agg[c]-df_agg[c.replace('_max', '_min')]
        except:
            pass
    
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
