#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:42:23 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils, os
utils.start(__file__)
#==============================================================================

PREF = 'f731_'

os.system(f'rm ../feature/t*_{PREF}*')


X = pd.read_pickle('../data/X_train_nejumi_gp.pkl.gz')
utils.to_feature(X.add_prefix(PREF), '../feature/train')


X = pd.read_pickle('../data/X_test_nejumi_gp.pkl.gz')
utils.to_feature(X.add_prefix(PREF), '../feature/test')



#==============================================================================
utils.end(__file__)


