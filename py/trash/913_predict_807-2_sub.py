#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:49:27 2018

@author: Kazuki
"""

import pandas as pd
from glob import glob
import utils


SUBMIT_FILE_PATH = f'../output/807-2.csv.gz'

COMMENT = f'CV805_LB803 (seed87 loop100)'

EXE_SUBMIT = True

label_name = 'TARGET'

# =============================================================================
# 
# =============================================================================
files = glob('../output/807-2*gz')

for i, file in enumerate(files):
    print(file)
    if i==0:
        sub = pd.read_csv(file)
        sub.TARGET = sub.TARGET.rank()
    else:
        sub.TARGET += pd.read_csv(file).TARGET.rank()
sub[label_name] /= len(files)
sub[label_name] /= sub[label_name].max()
sub['SK_ID_CURR'] = sub['SK_ID_CURR'].map(int)


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)

