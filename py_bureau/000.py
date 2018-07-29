#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:41:45 2018

@author: Kazuki
"""


import pandas as pd
import os
import utils
utils.start(__file__)
# =============================================================================

folders = ['../feature_bureau', '../feature_bureau_unused']


for fol in folders:
    os.system(f'rm -rf {fol}')
    os.system(f'mkdir {fol}')



train = utils.load_train(['SK_ID_CURR', 'TARGET'])
test = utils.load_test(['SK_ID_CURR'])

bureau = utils.read_pickles('../data/bureau')


bureau_train = pd.merge(bureau, train, on='SK_ID_CURR', how='inner')
bureau_test  = pd.merge(bureau, test, on='SK_ID_CURR', how='inner')



utils.to_pickles(bureau_train, '../data/bureau_train', utils.SPLIT_SIZE)
utils.to_pickles(bureau_test, '../data/bureau_test', utils.SPLIT_SIZE)

utils.to_pickles(bureau_train[['TARGET']], '../data/bureau_label', utils.SPLIT_SIZE)

"""

bureau_train = utils.read_pickles('../data/bureau_train')
bureau_test  = utils.read_pickles('../data/bureau_test')

"""


#==============================================================================
utils.end(__file__)
