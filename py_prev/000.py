#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:22:58 2018

@author: Kazuki
"""

import pandas as pd
import os
import utils
utils.start(__file__)
# =============================================================================

folders = ['../feature_prev', '../feature_prev_unused']


for fol in folders:
    os.system(f'rm -rf {fol}')
    os.system(f'mkdir {fol}')



train = utils.load_train(['SK_ID_CURR', 'TARGET'])
test = utils.load_test(['SK_ID_CURR'])

prev = utils.read_pickles('../data/previous_application')


prev_train = pd.merge(prev, train, on='SK_ID_CURR', how='inner')
prev_test  = pd.merge(prev, test, on='SK_ID_CURR', how='inner')



utils.to_pickles(prev_train, '../data/prev_train', utils.SPLIT_SIZE)
utils.to_pickles(prev_test, '../data/prev_test', utils.SPLIT_SIZE)

utils.to_pickles(prev_train[['TARGET']], '../data/prev_label', utils.SPLIT_SIZE)

"""

prev_train = utils.read_pickles('../data/prev_train')
prev_test  = utils.read_pickles('../data/prev_test')

"""



#==============================================================================
utils.end(__file__)


