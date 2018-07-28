#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:41:21 2018

@author: Kazuki
"""

from tqdm import tqdm
import pandas as pd
#import numpy as np


#def load_best_train():
#    files = ('../feature/train_' + pd.Series(feature_722) + '.f').tolist()
#    X = pd.concat([
#                    pd.read_feather(f) for f in tqdm(files, mininterval=60)
#                   ], axis=1)
#    return X
#
#def load_best_test():
#    files = ('../feature/test_' + pd.Series(feature_722) + '.f').tolist()
#    X = pd.concat([
#                    pd.read_feather(f) for f in tqdm(files, mininterval=60)
#                   ], axis=1)
#    return X

# =============================================================================
# LB804
# =============================================================================
def load_train_lb804():
    return pd.read_feather('../data/X_train_LB0.804.f')

def load_test_lb084():
    return pd.read_feather('../data/X_test_LB0.804.f')


# =============================================================================
# LB804
# =============================================================================
def load_train_lb806():
    return pd.read_pickle('../feature_someone/0727/20180727_train_rk.pkl')

def load_test_lb086():
    return pd.read_pickle('../feature_someone/0727/20180727_test_rk.pkl')

def load_cat_lb806():
    return ['FLAG_DOCUMENT_PATTERN' 'FLAG_OWN_CAR' 'FLAG_OWN_REALTY'
         'FONDKAPREMONT_MODE' 'HOUSETYPE_MODE' 'NAME_CONTRACT_TYPE'
         'NAME_EDUCATION_TYPE' 'NAME_FAMILY_STATUS' 'NAME_HOUSING_TYPE'
         'NAME_INCOME_TYPE' 'NAME_TYPE_SUITE' 'OCCUPATION_TYPE'
         'WALLSMATERIAL_MODE' 'WEEKDAY_APPR_PROCESS_START']



