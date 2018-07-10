#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 08:53:06 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
#import utils_agg
import utils
#utils.start(__file__)
#==============================================================================

files = utils.get_use_files([], True)
X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)


# =============================================================================
# var0
# =============================================================================
col_var0 = utils.check_var(X)

def multi_touch_var0(arg):
    os.system(f'touch "../feature_var0/{arg}.f"')

pool = Pool(cpu_count())
pool.map(multi_touch_var0, col_var0)
pool.close()


# =============================================================================
# corr1
# =============================================================================
col_corr1 = utils.check_corr(df_agg, corr_limit=.98, sample_size=19999)

def multi_touch_corr1(arg):
    os.system(f'touch "../feature_corr1/{arg}.f"')

pool = Pool(cpu_count())
pool.map(multi_touch_corr1, col_corr1)
pool.close()




#==============================================================================
utils.end(__file__)
