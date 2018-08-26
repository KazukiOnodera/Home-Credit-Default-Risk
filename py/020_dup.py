#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:49:00 2018

@author: Kazuki
"""

import pandas as pd
import numpy as np
import utils, os
#utils.start(__file__)
#==============================================================================
PREF = 'f020_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
#==============================================================================


dup = pd.read_csv('../data/same_user_all.csv.gz')
feature = dup[['SK_ID_CURR']].set_index('SK_ID_CURR')

dup.loc[dup['DAYS_EMPLOYED']==365243, 'DAYS_EMPLOYED'] = np.nan

dup['seq'] = 1
dup['seq'] = dup.groupby('dup_id').seq.cumsum()-1

category = dup.select_dtypes('O').columns
dup = pd.get_dummies(dup, columns=category)


gr = dup.groupby('dup_id')

col = dup.head().drop(['dup_id', 'dup_cnt'], axis=1).columns.tolist()

# last
for c in col[1:]:
    feature[f'last_{c}'] = gr[c].shift().values

# diff
for c in col[2:]:
    feature[f'{c}_diff'] = gr[c].diff().values

# ratio
for c in col[2:]:
    feature[f'{c}_ratio'] = ( dup[c] / gr[c].shift() ).values

# min
for c in col[2:]:
    feature[f'{c}_min'] = pd.concat([ dup[c], gr[c].shift()], axis=1).min(1).values

# max
for c in col[2:]:
    feature[f'{c}_max'] = pd.concat([ dup[c], gr[c].shift()], axis=1).max(1).values

# mean
for c in col[2:]:
    feature[f'{c}_mean'] = pd.concat([ dup[c], gr[c].shift()], axis=1).mean(1).values



feature.dropna(how='all', inplace=True)

utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)


train = utils.load_train([KEY])
test = utils.load_test([KEY])

feature.reset_index(inplace=True)
feature = pd.merge(feature, pd.read_csv('../data/user_id.csv.gz'), on=KEY, how='left')


tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature/train')

tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')




#==============================================================================
#utils.end(__file__)





