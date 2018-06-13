#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 00:23:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================

PREF = 'prev_101_'

KEY = 'SK_ID_CURR'

# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

gr = prev.groupby(KEY)

base = gr['remain_debt'].sum()
base.name = 'remain_debt_sum'
base[''] = gr.size()




base = base.reset_index()
# =============================================================================
# 
# =============================================================================

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1).fillna(0)
utils.to_feature(train.add_prefix(PREF), '../feature/train')


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1).fillna(0)
utils.to_feature(test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)


