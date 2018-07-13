#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:07:28 2018

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
PREF = 'f305_'

KEY = 'SK_ID_PREV'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature_prev/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

prev = utils.read_pickles('../data/previous_application', ['SK_ID_PREV', 'NAME_CONTRACT_TYPE'])

train = utils.read_pickles('../data/prev_train', [KEY])
test  = utils.read_pickles('../data/prev_test', [KEY])

# =============================================================================
# 
# =============================================================================
def multi_agg(args):
    path, pref, cont_type, cont_type_pref = args
    print(args)
    
    ins = utils.read_pickles(path)
    ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]
    ins = pd.merge(ins, prev, on='SK_ID_PREV', how='left'); gc.collect()
    del ins['SK_ID_CURR']
    
    
    
    if cont_type=='NA':
        df = ins[ins['NAME_CONTRACT_TYPE'].isnull()]
    else:
        df = ins[ins['NAME_CONTRACT_TYPE']==cont_type]
    
    df_agg = df.groupby(KEY).agg({**utils_agg.ins_num_aggregations})
    
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['INS_COUNT'] = df.groupby(KEY).size()
    df_agg = df_agg.add_prefix(pref+cont_type_pref).reset_index()
    
    utils.remove_feature(df_agg, var_limit=0, sample_size=19999)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature_prev/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature_prev/test')
    
    return

# =============================================================================
# main
# =============================================================================
argss = [
        ['../data/installments_payments', '', 'Consumer loans', 'con_'],
        ['../data/installments_payments', '', 'Cash loans', 'cas_'],
        ['../data/installments_payments', '', 'Revolving loans', 'rev_'],
        ['../data/installments_payments', '', 'NA', 'nan_'],
        ]
argss += [
        ['../data/installments_payments_delay', 'delay_', 'Consumer loans', 'con_'],
        ['../data/installments_payments_delay', 'delay_', 'Cash loans', 'cas_'],
        ['../data/installments_payments_delay', 'delay_', 'Revolving loans', 'rev_'],
        ['../data/installments_payments_delay', 'delay_', 'NA', 'nan_'],
        ]
argss += [
        ['../data/installments_payments_notdelay', 'notdelay_', 'Consumer loans', 'con_'],
        ['../data/installments_payments_notdelay', 'notdelay_', 'Cash loans', 'cas_'],
        ['../data/installments_payments_notdelay', 'notdelay_', 'Revolving loans', 'rev_'],
        ['../data/installments_payments_notdelay', 'notdelay_', 'NA', 'nan_'],
        ]

pool = Pool(len(argss))
pool.map(multi_agg, argss)
pool.close()


#==============================================================================
utils.end(__file__)
