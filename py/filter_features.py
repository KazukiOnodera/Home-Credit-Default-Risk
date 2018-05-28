#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:14:31 2018

@author: Kazuki
"""


from glob import glob
import pandas as pd
from time import sleep
import os
import gc
import utils
utils.start(__file__)
#==============================================================================

fname = 'LOG/imp_901_cv_527-1.py.csv'


# =============================================================================
# 
# =============================================================================
imp = pd.read_csv(fname).set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()

folders = sorted(glob('../data/*_train'))

def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder, utils.SPLIT_SIZE)
        
        df = utils.read_pickles(folder.replace('_train', '_test'), col)
        utils.to_pickles(df, folder, utils.SPLIT_SIZE)
    else:
        print(f'{folder} doesnt have valid features')
        pass
    


#==============================================================================
utils.end(__file__)
