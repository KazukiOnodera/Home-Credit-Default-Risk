#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:32:11 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f110_'

KEY = 'SK_ID_CURR'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/previous_application')
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True, ascending=[True, False])

col_num = [c for c in prev.columns[2:] if prev[c].dtype!='O']


prev.groupby(KEY)[col_num].pct_change(-1)

