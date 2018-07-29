#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 01:40:35 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'f705_'

KEY = 'SK_ID_CURR'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# load
# =============================================================================
train = utils.load_train([KEY]).set_index(KEY)
test = utils.load_test([KEY]).set_index(KEY)


prev_train = pd.read_feather('../data/prev_train_imputation_f705.f')
prev_test  = pd.read_feather('../data/prev_test_imputation_f705.f')


# =============================================================================
# feature
# =============================================================================
# train
gr = prev_train.groupby(KEY)
train['prev_y_min'] = gr.y_pred.min()
train['prev_y_mean'] = gr.y_pred.mean()
train['prev_y_max'] = gr.y_pred.max()
train['prev_y_var'] = gr.y_pred.var()
train['prev_y_median'] = gr.y_pred.median()
train['prev_y_q25'] = gr.y_pred.quantile(.25)
train['prev_y_q75'] = gr.y_pred.quantile(.75)

# test
gr = prev_test.groupby(KEY)
test['prev_y_min'] = gr.y_pred.min()
test['prev_y_mean'] = gr.y_pred.mean()
test['prev_y_max'] = gr.y_pred.max()
test['prev_y_var'] = gr.y_pred.var()
test['prev_y_median'] = gr.y_pred.median()
test['prev_y_q25'] = gr.y_pred.quantile(.25)
test['prev_y_q75'] = gr.y_pred.quantile(.75)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# =============================================================================
# output
# =============================================================================
utils.to_feature(train.add_prefix(PREF), '../feature/train')
utils.to_feature(test.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)


