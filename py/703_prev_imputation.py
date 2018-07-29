#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 01:36:21 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f703_'

KEY = 'SK_ID_CURR'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
train = utils.load_train([KEY]).set_index(KEY)
test = utils.load_test([KEY]).set_index(KEY)


bureau_train = pd.read_feather('../data/bureau_train_imputation_f703.f')
bureau_test  = pd.read_feather('../data/bureau_test_imputation_f703.f')


# =============================================================================
# feature
# =============================================================================
# train
gr = bureau_train.groupby(KEY)
train['bureau_y_min'] = gr.y_pred.min()
train['bureau_y_mean'] = gr.y_pred.mean()
train['bureau_y_max'] = gr.y_pred.max()
train['bureau_y_var'] = gr.y_pred.var()
train['bureau_y_median'] = gr.y_pred.median()
train['bureau_y_q25'] = gr.y_pred.quantile(.25)
train['bureau_y_q75'] = gr.y_pred.quantile(.75)

# test
gr = bureau_test.groupby(KEY)
test['bureau_y_min'] = gr.y_pred.min()
test['bureau_y_mean'] = gr.y_pred.mean()
test['bureau_y_max'] = gr.y_pred.max()
test['bureau_y_var'] = gr.y_pred.var()
test['bureau_y_median'] = gr.y_pred.median()
test['bureau_y_q25'] = gr.y_pred.quantile(.25)
test['bureau_y_q75'] = gr.y_pred.quantile(.75)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# =============================================================================
# output
# =============================================================================
utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)




