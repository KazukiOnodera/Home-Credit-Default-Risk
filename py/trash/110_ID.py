#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:48:18 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f110_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# load
# =============================================================================
train = utils.load_train(['SK_ID_CURR']).set_index('SK_ID_CURR')
test = utils.load_test(['SK_ID_CURR']).set_index('SK_ID_CURR')
prev = utils.read_pickles('../data/previous_application', ['SK_ID_CURR', 'SK_ID_PREV'])

# =============================================================================
# prev
# =============================================================================
gr = prev.groupby('SK_ID_CURR')

train['SK_ID_PREV_min'] = gr.SK_ID_PREV.min()
train['SK_ID_PREV_mean'] = gr.SK_ID_PREV.mean()
train['SK_ID_PREV_max'] = gr.SK_ID_PREV.max()
train['SK_ID_PREV_median'] = gr.SK_ID_PREV.median()
train['SK_ID_PREV_std'] = gr.SK_ID_PREV.std()
train['SK_ID_PREV_std-d-mean'] = train['SK_ID_PREV_std'] / train['SK_ID_PREV_mean']
train['SK_ID_PREV_max-m-min'] = train['SK_ID_PREV_max'] - train['SK_ID_PREV_min']

test['SK_ID_PREV_min'] = gr.SK_ID_PREV.min()
test['SK_ID_PREV_mean'] = gr.SK_ID_PREV.mean()
test['SK_ID_PREV_max'] = gr.SK_ID_PREV.max()
test['SK_ID_PREV_median'] = gr.SK_ID_PREV.median()
test['SK_ID_PREV_std'] = gr.SK_ID_PREV.std()
test['SK_ID_PREV_std-d-mean'] = test['SK_ID_PREV_std'] / test['SK_ID_PREV_mean']
test['SK_ID_PREV_max-m-min'] = test['SK_ID_PREV_max'] - test['SK_ID_PREV_min']

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# =============================================================================
# output
# =============================================================================
utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)


