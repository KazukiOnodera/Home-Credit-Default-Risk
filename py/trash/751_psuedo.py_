#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 23:29:23 2018

@author: Kazuki
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils, utils_best

# =============================================================================

os.system('rm -rf ../psuedo')
os.system('mkdir ../psuedo')

THRESHOLD = 0.036

SEED = 71

np.random.seed(SEED)
# =============================================================================



loader = utils_best.Loader('LB804')


X_test = loader.test()
prob = pd.read_csv('../data/LB804_test_pred2.csv').y_pred_lr

X_test['prob'] = prob

def sampling(threshold, i):
    df = X_test[X_test['prob'] < threshold]
    df['TARGET'] = (df['prob'] > np.random.uniform(size=df.shape[0]))*1
    del df['prob']
    print(df['TARGET'].mean())
    df.reset_index(drop=True).to_feather(f'../psuedo/{i}.f')
    return

[sampling(THRESHOLD, i) for i in range(10)]


