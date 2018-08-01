#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:56:44 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
#import sys
#sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
#import lgbextension as ex
#import lightgbm as lgb
#from multiprocessing import cpu_count
#from glob import glob
import utils, utils_cat

# =============================================================================
# setting
# =============================================================================

filename_in = '../output/730-1_X.csv.gz'

filename_out = 'CV805_LB803'

# =============================================================================
# load df
# =============================================================================

df = pd.read_csv(filename_in)



# =============================================================================
# load train
# =============================================================================

files = ('../feature/train_' + df.columns + '.f').tolist()
#files = utils.get_use_files(files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)

X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')
gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))
print(f'category: {CAT}')

COL = X.columns.tolist()

# =============================================================================
# test
# =============================================================================
files = ('../feature/test_' + df.columns + '.f').tolist()

dtest = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)

dtest['nejumi'] = np.load('../feature_someone/test_nejumi.npy')
dtest = dtest[COL]

# =============================================================================
# def
# =============================================================================
def reduce_mem_usage(df):
    col_int8 = []
    col_int16 = []
    col_int32 = []
    col_int64 = []
    col_float16 = []
    col_float32 = []
    col_float64 = []
    col_cat = []
    for c in tqdm(df.columns, mininterval=20):
        col_type = df[c].dtype

        if col_type != object:
            c_min = df[c].min()
            c_max = df[c].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    col_int8.append(c)
                    
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    col_int16.append(c)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    col_int32.append(c)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    col_int64.append(c)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    col_float16.append(c)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    col_float32.append(c)
                else:
                    col_float64.append(c)
        else:
            col_cat.append(c)
    
    if len(col_int8)>0:
        df[col_int8] = df[col_int8].astype(np.int8)
    if len(col_int16)>0:
        df[col_int16] = df[col_int16].astype(np.int16)
    if len(col_int32)>0:
        df[col_int32] = df[col_int32].astype(np.int32)
    if len(col_int64)>0:
        df[col_int64] = df[col_int64].astype(np.int64)
    if len(col_float16)>0:
        df[col_float16] = df[col_float16].astype(np.float16)
    if len(col_float32)>0:
        df[col_float32] = df[col_float32].astype(np.float32)
    if len(col_float64)>0:
        df[col_float64] = df[col_float64].astype(np.float64)
    if len(col_cat)>0:
        df[col_cat] = df[col_cat].astype('category')


def to_pkl_gzip(df, path):
    df.to_pickle(path)
    os.system('gzip ' + path)
    return

# =============================================================================
# align
# =============================================================================
X_train, X_test = X.align(dtest, join='inner', axis=1)

reduce_mem_usage(X_train)
reduce_mem_usage(X_test)

print(f'train.shape: {X_train.shape}, test.shape: {X_test.shape}')

# feather
#X_train.to_feather(f'../data/X_train_{filename_out}.f')
#X_test.to_feather(f'../data/X_test_{filename_out}.f')

# =============================================================================
# pickle
# =============================================================================
X_train.to_pickle(f'../data/X_train_{filename_out}.pkl')
X_test.to_pickle(f'../data/X_test_{filename_out}.pkl')

os.system(f'gzip ../data/X_train_{filename_out}.pkl')
os.system(f'gzip ../data/X_test_{filename_out}.pkl')

to_pkl_gzip(X_train, f'../data/X_train_{filename_out}.pkl')
to_pkl_gzip(X_test, f'../data/X_test_{filename_out}.pkl')

