#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:30:32 2018

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

imp_name = 'LOG/imp_110_pos-2-cv.py.csv'

folders = glob('../data/108*_train')
folders += glob('../data/109*_train')
folders += glob('../data/110*_train')

# =============================================================================
# 
# =============================================================================
imp = pd.read_csv(imp_name).set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()


def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        del df; gc.collect()
        
        folder = folder.replace('_train', '_test')
        df = utils.read_pickles(folder, col)
        utils.to_pickles(df, folder+'_filtered', utils.SPLIT_SIZE)
        
    else:
        print(f'{folder} doesnt have valid features')
        pass
    

[read_pickle(f, feature_all) for f in folders]

#==============================================================================
utils.end(__file__)


