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
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
#import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from glob import glob
import utils, utils_cat

# =============================================================================
# 
# =============================================================================

FEATURE_SIZE = 600


# =============================================================================
# load train
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)

files = ('../feature/train_' + imp.head(FEATURE_SIZE).feature + '.f').tolist()
#files = utils.get_use_files(files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

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
files = ('../feature/test_' + imp.head(FEATURE_SIZE).feature + '.f').tolist()

dtest = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)

dtest['nejumi'] = np.load('../feature_someone/test_nejumi.npy')
dtest = dtest[COL]

# =============================================================================
# 
# =============================================================================
X_train, X_test = X.align(dtest, join='inner', axis=1)


print(f'train.shape: {X_train.shape}, test.shape: {X_test.shape}')

X_train.to_feather('../data/X_train_LB0.804.f')
X_test.to_feather('../data/X_test_LB0.804.f')


# =============================================================================
# 
# =============================================================================
tmp = pd.read_csv('../output/725-2_X.csv.gz')

print(tmp.columns.difference(X_train.columns))
print(X_train.columns.difference(tmp.columns))
print(tmp.columns.difference(X_test.columns))

