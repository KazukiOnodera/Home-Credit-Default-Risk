#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:18:28 2018

@author: Kazuki

bureau Active

"""

import os
import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
from glob import glob
import utils
#utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'bureau_active'
NTHREAD = 3


col_num = ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
           'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
           'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
           'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

col_group = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

# =============================================================================
# feature
# =============================================================================
bureau = utils.read_pickles('../data/bureau')
bureau = bureau[bureau['CREDIT_ACTIVE']=='Active']

base = bureau[[KEY]].drop_duplicates().set_index(KEY)

train = utils.load_train([KEY])
test = utils.load_test([KEY])

def nunique(x):
    return len(set(x))

# =============================================================================
# gr2
# =============================================================================
def multi_gr2(k):
    gr2 = bureau.groupby([KEY, k])
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
    base.to_pickle(f'../data/tmp_504-gr2_{PREF}_{k}.p')
    
pool = Pool(NTHREAD)
callback = pool.map(multi_gr2, col_group)
pool.close()

# =============================================================================
# pivot
# =============================================================================
def pivot(cat):
    li = []
    pt = pd.pivot_table(bureau, index=KEY, columns=cat, values=col_num)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_mean'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(bureau, index=KEY, columns=cat, values=col_num, aggfunc=np.sum)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_sum'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    pt = pd.pivot_table(bureau, index=KEY, columns=cat, values=col_num, aggfunc=np.std, fill_value=-1)
    pt.columns = [f'{PREF}_{cat}_{c[0]}-{c[1]}_std'.replace(' ', '-') for c in pt.columns]
    li.append(pt)
    feat = pd.concat(li, axis=1).reset_index()
    del li, pt
    gc.collect()
    
    df = pd.merge(train, feat, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/tmp_504_{cat}_train', utils.SPLIT_SIZE)
    gc.collect()
    
    df = pd.merge(test, feat, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df,  f'../data/tmp_504_{cat}_test',  utils.SPLIT_SIZE)
    gc.collect()

pool = Pool(NTHREAD)
callback = pool.map(pivot, col_cat)
pool.close()

# =============================================================================
# gr1
# =============================================================================
gr = bureau.groupby(KEY)

# stats
keyname = 'gby-'+KEY
base[f'{PREF}_{keyname}_size'] = gr.size()
for c in col_num:
    gc.collect()
    print(c)
    base[f'{PREF}_{keyname}_{c}_min'] = gr[c].min()
    base[f'{PREF}_{keyname}_{c}_max'] = gr[c].max()
    base[f'{PREF}_{keyname}_{c}_max-min'] = base[f'{PREF}_{keyname}_{c}_max'] - base[f'{PREF}_{keyname}_{c}_min']
    base[f'{PREF}_{keyname}_{c}_mean'] = gr[c].mean()
    base[f'{PREF}_{keyname}_{c}_std'] = gr[c].std()
    base[f'{PREF}_{keyname}_{c}_sum'] = gr[c].sum()
    base[f'{PREF}_{keyname}_{c}_nunique'] = gr[c].apply(nunique)

'bureau_active_gby-SK_ID_CURR_AMT_CREDIT_SUM_DEBT_sum'

# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(bureau[KEY], bureau[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(bureau[KEY], bureau[c1], normalize='index')
    df_norm.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
    df = pd.concat([df_sum, df_norm], axis=1)
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)

# =============================================================================
# merge
# =============================================================================
df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(f'../data/tmp_504-gr2_{PREF}*.p'))], axis=1)
base = pd.concat([base, df], axis=1)
base.reset_index(inplace=True)
del df; gc.collect()
#base.columns = [c.replace('bureau_active__', 'bureau_active_') for c in base.columns]


train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
train_ = pd.concat([utils.read_pickles(f) for f in sorted(glob(f'../data/tmp_504_*_train'))], axis=1)
train = pd.concat([train, train_], axis=1)
utils.to_pickles(train, '../data/504_train', utils.SPLIT_SIZE)
del train; gc.collect()


test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
test_ = pd.concat([utils.read_pickles(f) for f in sorted(glob(f'../data/tmp_504_*_test'))], axis=1)
test = pd.concat([test, test_], axis=1)
utils.to_pickles(test,  '../data/504_test',  utils.SPLIT_SIZE)


os.system('rm -rf ../data/tmp_504*')

#==============================================================================
utils.end(__file__)


