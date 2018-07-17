#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:09:29 2018

@author: Kazuki
"""


import pandas as pd
import numpy as np
import os
import utils
#utils.start(__file__)
# =============================================================================
PREF = 'irk_'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================

tr = pd.DataFrame(np.load('../feature_someone/train_ireko.npy'), columns=['ireko'])
te = pd.DataFrame(np.load('../feature_someone/test_ireko.npy'), columns=['ireko'])




if tr.shape[1] != te.shape[1]:
    raise Exception('unmatch')

if not len(tr.columns.difference(te.columns)) == len(te.columns.difference(tr.columns)) == 0:
    raise


#tr.to_feather('../feature_someone/Maxwell_train.f')
#te.to_feather('../feature_someone/Maxwell_test.f')

utils.to_feature(tr.add_prefix(PREF), '../feature/train')
utils.to_feature(te.add_prefix(PREF), '../feature/test')





#==============================================================================
utils.end(__file__)

