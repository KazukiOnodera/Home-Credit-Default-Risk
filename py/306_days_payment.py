#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:57:48 2018

@author: Kazuki

支払う間隔

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
PREF = 'f306_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================

train = utils.load_train([KEY])
test = utils.load_test([KEY])

ins = utils.read_pickles('../data/installments_payments_delay')


# =============================================================================
# 
# =============================================================================

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def aggregate(args):
    path, pref = args
    
    df = utils.read_pickles(path, [KEY, 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']).drop_duplicates(['SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'])
    df = df[df['DAYS_ENTRY_PAYMENT'].between(day_start, day_end)].sort_values(['SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'])
    
    df['DEP_diff'] = df.groupby('SK_ID_PREV').DAYS_ENTRY_PAYMENT.diff()
    feature = df.groupby(KEY).agg({'DEP_diff': ['min', 'mean', 'max', 'var', 'nunique']})
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
argss = [
        ['../data/installments_payments', '']
        ['../data/installments_payments_delay', 'delay_']
        ['../data/installments_payments_notdelay', 'notdelay_']
        ]
pool = Pool(3)
pool.map(aggregate, argss)
pool.close()




#==============================================================================
utils.end(__file__)
