#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:02:39 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f156_'

KEY = 'SK_ID_CURR'

col_binary = ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON',
              'NAME_YIELD_GROUP', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 
              'NAME_PRODUCT_TYPE', 'NAME_SELLER_INDUSTRY', 'CHANNEL_TYPE',
              'NAME_PAYMENT_TYPE']

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/future_application', ['SK_ID_CURR', 'DAYS_DECISION']+col_binary)
#base = prev[[KEY]].drop_duplicates().set_index(KEY)

prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True) # top is latest

col_binary_di = {}

for c in col_binary:
    col_binary_di[c] = list(prev[c].unique())

ids = prev.SK_ID_CURR.unique()

def to_decimal(x):
    if len(x)==0:
        return -1
    x = ''.join(map(str, x))
    return float(x[0] + '.' + x[1:])


def multi(id_curr):
    """
    id_curr = 101043
    """
    tmp = prev[prev.SK_ID_CURR==id_curr]
    tmp_app = tmp[tmp['NAME_CONTRACT_STATUS']=='Approved']
    tmp_ref = tmp[tmp['NAME_CONTRACT_STATUS']=='Refused']
    tmp_appref = tmp[tmp['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])]
    di = {}
    try:
        for c in col_binary:
            for v in col_binary_di[c]:
                di[f'{c}-{v}']        = to_decimal( ((tmp[c]==v)*1).tolist() )
                di[f'{c}-{v}_app']    = to_decimal( ((tmp_app[c]==v)*1).tolist() )
                di[f'{c}-{v}_ref']    = to_decimal( ((tmp_ref[c]==v)*1).tolist() )
                di[f'{c}-{v}_appref'] = to_decimal( ((tmp_appref[c]==v)*1).tolist() )
    except:
        print(id_curr, c)
        raise
    tmp = pd.DataFrame.from_dict(di, orient='index').T
    tmp['SK_ID_CURR'] = id_curr
    return tmp.set_index('SK_ID_CURR')


# =============================================================================
# main
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi, ids)
pool.close()


base = pd.concat(callback)
utils.remove_feature(base)
# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)
if base.columns.duplicated().sum() != 0:
    raise Exception( base.columns[base.columns.duplicated()] )

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(train.add_prefix(PREF), '../feature/train')


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(test.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)
#utils.stop_instance()
