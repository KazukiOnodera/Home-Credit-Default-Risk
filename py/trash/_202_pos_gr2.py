#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:03 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'pos'

NTHREAD = 2

col_num = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
           'SK_DPD', 'SK_DPD_DEF']

col_cat = ['NAME_CONTRACT_STATUS']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
base = pos[[KEY]].drop_duplicates().set_index(KEY)

def nunique(x):
    return len(set(x))

def multi_gr2(k):
    gr2 = pos.groupby([KEY, k])
    gc.collect()
    print(k)
    keyname = 'gby-'+'-'.join([KEY, k])
    # size
    gr1 = gr2.size().groupby(KEY)
    name = f'{PREF}_{keyname}_size'
    base[f'{name}_min']  = gr1.min()
    base[f'{name}_max']  = gr1.max()
    base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
    base[f'{name}_mean'] = gr1.mean()
    base[f'{name}_std']  = gr1.std()
    base[f'{name}_sum']  = gr1.sum()
    base[f'{name}_nunique']     = gr1.size()
    for v in col_num:
        
        # min
        gr1 = gr2[v].min().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_min'
        base[f'{name}_max']     = gr1.max()
        base[f'{name}_mean']    = gr1.mean()
        base[f'{name}_std']     = gr1.std()
        base[f'{name}_sum']     = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # max
        gr1 = gr2[v].max().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_max'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # mean
        gr1 = gr2[v].mean().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_mean'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # std
        gr1 = gr2[v].std().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_std'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # sum
        gr1 = gr2[v].sum().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_sum'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_nunique'] = gr1.apply(nunique)
    base.to_pickle(f'../data/tmp_202_{PREF}_{k}.p')
    
# =============================================================================
# gr2
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi_gr2, col_group)
pool.close()

# =============================================================================
# merge
# =============================================================================
df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(f'../data/tmp_202_{PREF}*.p'))], axis=1)
base = pd.concat([base, df], axis=1)
base.reset_index(inplace=True)
del df; gc.collect()

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/202_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/202_test',  utils.SPLIT_SIZE)



os.system('rm ../data/tmp_202_*.p')


#==============================================================================
utils.end(__file__)


