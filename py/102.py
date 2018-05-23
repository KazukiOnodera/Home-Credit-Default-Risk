#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:19:26 2018

@author: kazuki.onodera

previous_application

"""

import numpy as np
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================


prev = utils.read_pickles('../data/previous_application')


df = pd.crosstab(prev['SK_ID_CURR'], prev['NAME_CONTRACT_TYPE'])

df.columns = ['prev_sum_'+c.replace(' ', '-') for c in df.columns]
df.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

key = 'SK_ID_CURR'
train = utils.load_train()[[key]]
train = pd.merge(train, df, on=key, how='left')


test = utils.load_test()[[key]]
test = pd.merge(test, df, on=key, how='left')

utils.to_pickles(train, '../data/102_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/102_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


