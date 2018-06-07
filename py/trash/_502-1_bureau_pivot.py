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
import os
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'bureau'


col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

NTHREAD = len(col_cat)

# =============================================================================
# pivot
# =============================================================================
base = utils.read_pickles('../data/bureau')
train = utils.load_train([KEY])
test = utils.load_test([KEY])

def pivot(cat):
    li = []
    pt = pd.pivot_table(base, index=KEY, columns=cat, values=col_num)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_mean'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(base, index=KEY, columns=cat, values=col_num, aggfunc=np.sum)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_sum'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(base, index=KEY, columns=cat, values=col_num, aggfunc=np.std, fill_value=-1)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_std'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    feat = pd.concat(li, axis=1).reset_index()
    del li, pt
    gc.collect()
    
    df = pd.merge(train, feat, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/tmp_502-1-{cat}_train', utils.SPLIT_SIZE)
    gc.collect()
    
    df = pd.merge(test, feat, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df,  f'../data/tmp_502-1-{cat}_test',  utils.SPLIT_SIZE)
    gc.collect()

    
# =============================================================================
# 
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(pivot, col_cat)
pool.close()


# =============================================================================
# concat
# =============================================================================


train = pd.concat([utils.read_pickles(f) for f in sorted(glob(f'../data/tmp_502-1-*_train'))], axis=1)
utils.to_pickles(train, '../data/502-1_train', utils.SPLIT_SIZE)


test = pd.concat([utils.read_pickles(f) for f in sorted(glob(f'../data/tmp_502-1-*_test'))], axis=1)
utils.to_pickles(test,  '../data/502-1_test',  utils.SPLIT_SIZE)


os.system('rm -rf ../data/tmp_502-1-*')


#==============================================================================
utils.end(__file__)



