#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:17:25 2018

@author: Kazuki

bureau

"""


import pandas as pd
import numpy as np
import gc
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'bureau'

NTHREAD = 15

col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']


# =============================================================================
# pivot
# =============================================================================
prev = utils.read_pickles('../data/previous_application')
train = utils.load_train([KEY])
test = utils.load_test([KEY])

def pivot(cat):
    li = []
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_mean' for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num, aggfunc=np.sum)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_sum' for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(prev, index=KEY, columns=cat, values=col_num, aggfunc=np.std, fill_value=-1)
    pt.columns = [f'{PREF}_{c[0]}-{c[1]}_std' for c in pt.columns]
    li.append(pt)
    base = pd.concat(li, axis=1).reset_index()
    base.reset_index(inplace=True)
    del li, pt
    gc.collect()
    
    df = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/109_{cat}_train', utils.SPLIT_SIZE)
    gc.collect()
    
    df = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df,  f'../data/109_{cat}_test',  utils.SPLIT_SIZE)
    gc.collect()

    
# =============================================================================
# 
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(pivot, col_cat)
pool.close()


#==============================================================================
utils.end(__file__)



