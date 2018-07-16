#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 08:08:17 2018

@author: Kazuki
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import utils
utils.start(__file__)
# =============================================================================
PREF = 'Mxw_'


os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================

tr = pd.read_csv('../feature_someone/train_x_lgbm_all_v7.csv')
te = pd.read_csv('../feature_someone/test_x_lgbm_all_v7.csv')


col_cat = ['FLAG_PHONE_PATTERN', 'FLAG_DOC_PATTERN']
le = LabelEncoder()
for c in col_cat:
    tr[c].fillna('na dayo', inplace=True)
    te[c].fillna('na dayo', inplace=True)
    le.fit( tr[c].append(te[c]) )
    tr[c] = le.transform(tr[c])
    te[c]  = le.transform(te[c])


col_O = [c for c in tr.columns if tr[c].dtype=='O']
tr.drop(col_O, axis=1, inplace=True)
te.drop(col_O, axis=1, inplace=True)

if tr.shape[1] != te.shape[1]:
    raise Exception('unmatch')

if not len(tr.columns.difference(te.columns)) == len(te.columns.difference(tr.columns)) == 0:
    raise


#tr.to_feather('../feature_someone/Maxwell_train.f')
#te.to_feather('../feature_someone/Maxwell_test.f')

utils.to_feature(tr.add_prefix(PREF), '../feature/train')
utils.to_feature(te.add_prefix(PREF), '../feature/test')





#==============================================================================
utils.end(__file__)

