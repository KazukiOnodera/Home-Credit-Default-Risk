#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:46:05 2018

@author: Kazuki
"""

import pandas as pd
import gc
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import utils
utils.start(__file__)
#==============================================================================

KEY = 'SK_ID_CURR'
PREF = 'prev_108'


# =============================================================================
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

prev = prev[prev['DAYS_LAST_DUE_1ST_VERSION'].between(0, 365)]

prev = utils.get_dummies(prev)
prev.columns = [c.replace('/', '') for c in prev.columns]
prev.drop('SK_ID_PREV', axis=1, inplace=True)

base = prev[[KEY]].drop_duplicates().set_index(KEY)

gr = prev.groupby(KEY)

train = utils.load_train([KEY])

test = utils.load_test([KEY])



def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================

def multi(p):
    
    if p==0:
        feature = gr.size()
        feature.name = f'{PREF}_{KEY}_size'
        feature = feature.reset_index()
    elif p==1:
        feature = gr.min().add_prefix(f'{PREF}_').add_suffix('_min').reset_index()
    elif p==2:
        feature = gr.max().add_prefix(f'{PREF}_').add_suffix('_max').reset_index()
    elif p==3:
        feature = gr.mean().add_prefix(f'{PREF}_').add_suffix('_mean').reset_index()
    elif p==4:
        feature = gr.std().add_prefix(f'{PREF}_').add_suffix('_std').reset_index()
    elif p==5:
        feature = gr.sum().add_prefix(f'{PREF}_').add_suffix('_sum').reset_index()
    elif p==6:
        feature = gr.quantile(0.25).add_prefix(f'{PREF}_').add_suffix('_q25').reset_index()
    elif p==7:
        feature = gr.quantile(0.50).add_prefix(f'{PREF}_').add_suffix('_q50').reset_index()
    elif p==8:
        feature = gr.quantile(0.75).add_prefix(f'{PREF}_').add_suffix('_q75').reset_index()
    else:
        return
    
    train_ = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(train_, '../feature/train')
    
    test_ = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(test_,  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================
pool = Pool(10)
pool.map(multi, range(10))
pool.close()



#==============================================================================
utils.end(__file__)


