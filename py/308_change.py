#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:59:40 2018

@author: kazuki.onodera

pct change

"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
#import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f308_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

train = utils.load_train([KEY])
test = utils.load_test([KEY])
prev = utils.read_pickles('../data/previous_application', ['SK_ID_PREV', 'NAME_CONTRACT_TYPE'])

COL = ['AMT_PAYMENT', 'AMT_PAYMENT-d-app_AMT_INCOME_TOTAL', 'AMT_PAYMENT-d-app_AMT_CREDIT',
       'AMT_PAYMENT-d-app_AMT_ANNUITY', 'AMT_PAYMENT-d-app_AMT_GOODS_PRICE']
# =============================================================================
# 
# =============================================================================

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def aggregate(args):
    path, cont_type, pref = args
    
    df = utils.read_pickles(path, [KEY, 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']+COL)
    df = df[df['DAYS_ENTRY_PAYMENT'].between(day_start, day_end)].sort_values([KEY, 'DAYS_ENTRY_PAYMENT'])
    df = pd.merge(df, prev, on='SK_ID_PREV', how='left'); gc.collect()
    
    if cont_type=='NA':
        df = df[df['NAME_CONTRACT_TYPE'].isnull()]
    else:
        df = df[df['NAME_CONTRACT_TYPE']==cont_type]
    
    df.sort_values(['SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'], inplace=True)
    
    col_ = []
    for c in COL:
        df[f'{c}_diff'] = df[['SK_ID_PREV', c]].groupby('SK_ID_PREV')[c].diff()
        df[f'{c}_diff_diff'] = df[['SK_ID_PREV', f'{c}_diff']].groupby('SK_ID_PREV')[f'{c}_diff'].diff()
        df[f'{c}_pctchange'] = df[['SK_ID_PREV', c]].groupby('SK_ID_PREV')[c].pct_change()
        df[f'{c}_pctchange_pctchange'] = df[['SK_ID_PREV', f'{c}_pctchange']].groupby('SK_ID_PREV')[f'{c}_pctchange'].pct_change()
        col_ += [f'{c}_diff', f'{c}_diff_diff', f'{c}_pctchange', f'{c}_pctchange_pctchange']
    
    num_agg = {}
    for c in col_:
        num_agg[c] = ['min', 'mean', 'max', 'var']
    
    feature = df.groupby(KEY).agg(**num_agg)
    feature.columns = pd.Index([e[0] + "_" + e[1] for e in feature.columns.tolist()])
    feature.reset_index(inplace=True)
    
    utils.remove_feature(feature, var_limit=0, sample_size=19999)
    
    tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF+pref), '../feature/train')
    
    tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF+pref),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

paths = [('../data/installments_payments', ''),
         ('../data/installments_payments_delay', 'delay_'),
         ('../data/installments_payments', 'notdelay_')]

cont_types = [('Consumer loans', 'cons_'),
              ('Cash loans', 'cas_'),
              ('Revolving loans', 'rev_'),
              ('NA', 'nan_')]

argss = []
for p in paths:
    for c in cont_types:
        print(p, c)
        path, pref1 = p
        cont_type, pref2 = c
        argss.append( [path, cont_type, pref1+pref2] )

pool = Pool(len(argss))
pool.map(aggregate, argss)
pool.close()




#==============================================================================
utils.end(__file__)
