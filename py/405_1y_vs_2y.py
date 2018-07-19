#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:56:44 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f405_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# 
# =============================================================================

tr_1yr = pd.concat([pd.read_feather(f) for f in sorted(glob('../feature/tr*_f402_*'))], axis=1)
tr_2yr = pd.concat([pd.read_feather(f) for f in sorted(glob('../feature/tr*_f403_*'))], axis=1)
train  = pd.DataFrame(index=tr_1yr.index)

for c1 in tr_1yr.columns:
    c2 = c1.replace('f402_', 'f403_')
    try:
        s = tr_1yr[c1] / tr_2yr[c2]
        train[c1.replace('f402_', PREF)] = s
    except:
        pass

te_1yr = pd.concat([pd.read_feather(f) for f in sorted(glob('../feature/te*_f402_*'))], axis=1)
te_2yr = pd.concat([pd.read_feather(f) for f in sorted(glob('../feature/te*_f403_*'))], axis=1)
test   = pd.DataFrame(index=te_1yr.index)

for c1 in te_1yr.columns:
    c2 = c1.replace('f402_', 'f403_')
    try:
        s = te_1yr[c1] / te_2yr[c2]
        test[c1.replace('f402_', PREF)] = s
    except:
        pass

train2, test2 = train.align(test, join='inner', axis=1)

# =============================================================================
# 
# =============================================================================
utils.to_feature(train2, '../feature/train')
utils.to_feature(test2, '../feature/test')


#==============================================================================
utils.end(__file__)

