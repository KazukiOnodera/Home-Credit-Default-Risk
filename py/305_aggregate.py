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
#utils.start(__file__)
#==============================================================================
PREF = 'f305_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

prev = utils.read_pickles('../data/previous_application', ['SK_ID_PREV', 'NAME_CONTRACT_TYPE'])

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
#def aggregate(df, pref):
#    
#    del df['SK_ID_PREV']
#    
#    # Consumer loans
#    df_ = df[df['NAME_CONTRACT_TYPE']=='Consumer loans']
#    
#    df_agg = df_.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
#    
#    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
#    
#    df_agg['INS_COUNT'] = df_.groupby('SK_ID_CURR').size()
#    df_agg = df_agg.add_prefix(pref+'cns_').reset_index()
#    
#    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
#    
#    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
#    
#    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
#    
#    # Cash loans
#    df_ = df[df['NAME_CONTRACT_TYPE']=='Cash loans']
#    
#    df_agg = df_.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
#    
#    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
#    
#    df_agg['INS_COUNT'] = df_.groupby('SK_ID_CURR').size()
#    df_agg = df_agg.add_prefix(pref+'csh_').reset_index()
#    
#    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
#    
#    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
#    
#    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
#    
#    # Revolving loans
#    df_ = df[df['NAME_CONTRACT_TYPE']=='Revolving loans']
#    
#    df_agg = df_.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
#    
#    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
#    
#    df_agg['INS_COUNT'] = df_.groupby('SK_ID_CURR').size()
#    df_agg = df_agg.add_prefix(pref+'rev_').reset_index()
#    
#    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
#    
#    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
#    
#    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
#    
#    # NA
#    df_ = df[df['NAME_CONTRACT_TYPE'].isnull()]
#    
#    df_agg = df_.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
#    
#    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
#    
#    df_agg['INS_COUNT'] = df_.groupby('SK_ID_CURR').size()
#    df_agg = df_agg.add_prefix(pref+'nan_').reset_index()
#    
#    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
#    
#    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
#    
#    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
#    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
#    
#    return

def multi_agg(args):
    pref, cont_type, cont_type_pref = args
    print(args)
    if cont_type=='NA':
        df = ins[ins['NAME_CONTRACT_TYPE'].isnull()]
    else:
        df = ins[ins['NAME_CONTRACT_TYPE']==cont_type]
    
    df_agg = df.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
    
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['INS_COUNT'] = df.groupby('SK_ID_CURR').size()
    df_agg = df_agg.add_prefix(pref+cont_type_pref).reset_index()
    
    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return

# =============================================================================
# main
# =============================================================================
ins = utils.read_pickles('../data/installments_payments')
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]
ins = pd.merge(ins, prev, on='SK_ID_PREV', how='left'); gc.collect()
del ins['SK_ID_PREV']

argss = [
        ['', 'Consumer loans', 'con_'],
        ['', 'Cash loans', 'cas_'],
        ['', 'Revolving loans', 'rev_'],
        ['', 'NA', 'nan_'],
        ]
pool = Pool(4)
pool.map(multi_agg, argss)
pool.close()

del ins; gc.collect()




ins = utils.read_pickles('../data/installments_payments_delay')
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]
ins = pd.merge(ins, prev, on='SK_ID_PREV', how='left'); gc.collect()

argss = [
        ['delay_', 'Consumer loans', 'con_'],
        ['delay_', 'Cash loans', 'cas_'],
        ['delay_', 'Revolving loans', 'rev_'],
        ['delay_', 'NA', 'nan_'],
        ]
pool = Pool(4)
pool.map(multi_agg, argss)
pool.close()

del ins; gc.collect()




ins = utils.read_pickles('../data/installments_payments_notdelay')
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]
ins = pd.merge(ins, prev, on='SK_ID_PREV', how='left'); gc.collect()

argss = [
        ['notdelay_', 'Consumer loans', 'con_'],
        ['notdelay_', 'Cash loans', 'cas_'],
        ['notdelay_', 'Revolving loans', 'rev_'],
        ['notdelay_', 'NA', 'nan_'],
        ]
pool = Pool(4)
pool.map(multi_agg, argss)
pool.close()



#==============================================================================
utils.end(__file__)
