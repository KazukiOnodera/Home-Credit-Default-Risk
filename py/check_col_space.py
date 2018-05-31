#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:46:09 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from glob import glob

folders = glob('../data/*_train')

def read(folder):
    df = pd.read_pickle(folder+'/000.p')
#    if 'prev_AMT_APPLICATION-XAP_mean' in df.columns:
#        print(folder)
    for c in df.columns:
        if ' ' == c:
            print(folder)
            break
    if df.columns.duplicated().sum()>0:
        print(f'duplicated: {folder} {df.columns[df.columns.duplicated()]}')
    return


[read(f) for f in folders]


