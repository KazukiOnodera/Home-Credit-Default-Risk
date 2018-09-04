#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 23:58:07 2018

@author: Kazuki

Onodera + Giba

"""


import pandas as pd
import numpy as np
import utils, os
#utils.start(__file__)
#==============================================================================
PREF = 'f025_'

path_user_id = '../data/user_id_v8.csv.gz'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
#==============================================================================

user_id = pd.read_csv(path_user_id)

train  = pd.read_csv('../input/application_train.csv.zip')
train['is_train'] = 1

test  = pd.read_csv('../input/application_test.csv.zip')
test['is_train'] = 0

trte = pd.concat([train, test], ignore_index=True)

dup = user_id[user_id.duplicated('user_id', False)]
cnt = dup.groupby('user_id').size()
cnt.name = 'dup_cnt'
cnt = cnt.reset_index()
cnt['dup_id'] = range(cnt.shape[0])

dup = pd.merge(dup, cnt, on='user_id', how='left')
dup = pd.merge(dup, trte, on=KEY, how='left')

col = train.columns.tolist() + [c for c in dup.columns if c not in train.columns]

dup = dup[col]


dup.loc[dup['DAYS_EMPLOYED']==365243, 'DAYS_EMPLOYED'] = np.nan

dup['seq'] = 1
dup['seq'] = dup.groupby('dup_id').seq.cumsum()-1

category = dup.select_dtypes('O').columns
dup = pd.get_dummies(dup, columns=category)


col = dup.columns.tolist()
for c in ['dup_id', 'dup_cnt', 'user_id']:
    col.remove(c)
col = ['user_id'] + col


# =============================================================================
# 
# =============================================================================
dup.sort_values(['user_id', 'DAYS_BIRTH'], ascending=[True, False], inplace=True)
dup = dup[dup.groupby('dup_id')['DAYS_BIRTH'].diff()!=0]

dup = dup[dup.duplicated('user_id', False)]

"""
feature['is_train'] = dup.is_train.values
feature['user_id'] = dup.user_id.values
feature['TARGET'] = dup.TARGET.values

feature.groupby('is_train').last_TARGET.describe()
feature.groupby('is_train').next_TARGET.describe()

feature[feature['is_train']==0][~feature.next_TARGET.isnull()].index

feature[['SK_ID_CURR', 'user_id', 'is_train', 'TARGET', 'last_TARGET']]
feature[feature.SK_ID_CURR.isin(drop_ids)][['SK_ID_CURR', 'user_id', 'is_train', 'TARGET', 'last_TARGET']]
feature[feature.SK_ID_CURR.isin(drop_ids)][ 'last_TARGET']
feature[~feature.SK_ID_CURR.isin(drop_ids)][ 'last_TARGET']


"""

feature = dup[['SK_ID_CURR']].set_index('SK_ID_CURR')
gr = dup.groupby('dup_id')


# last
for c in col[2:]:
    feature[f'last_{c}'] = gr[c].shift(1).values
#    feature[f'lastlast_{c}'] = gr[c].shift(2).values
    
#    feature[f'next_{c}'] = gr[c].shift(-1).values
#    feature[f'nextnext_{c}'] = gr[c].shift(-2).values

# other
for c in col[3:]:
    feature[f'{c}_diff'] = gr[c].diff(1).values
    feature[f'{c}_ratio'] = ( dup[c] / gr[c].shift(1) ).values
    feature[f'{c}_min'] = pd.concat([ dup[c], gr[c].shift(1), gr[c].shift(2)], axis=1).min(1).values
    feature[f'{c}_max'] = pd.concat([ dup[c], gr[c].shift(1), gr[c].shift(2)], axis=1).max(1).values
    feature[f'{c}_mean'] = pd.concat([ dup[c], gr[c].shift(1), gr[c].shift(2)], axis=1).mean(1).values
    
#    feature[f'{c}_diff_r'] = gr[c].diff(-1).values
#    feature[f'{c}_ratio_r'] = ( dup[c] / gr[c].shift(-1) ).values
#    feature[f'{c}_min_r'] = pd.concat([ dup[c], gr[c].shift(-1), gr[c].shift(-2)], axis=1).min(1).values
#    feature[f'{c}_max_r'] = pd.concat([ dup[c], gr[c].shift(-1), gr[c].shift(-2)], axis=1).max(1).values
#    feature[f'{c}_mean_r'] = pd.concat([ dup[c], gr[c].shift(-1), gr[c].shift(-2)], axis=1).mean(1).values


feature.dropna(how='all', inplace=True)
utils.remove_feature(feature, var_limit=0, corr_limit=0.98, sample_size=19999)


train = utils.load_train([KEY])
test = utils.load_test([KEY])

feature.reset_index(inplace=True)
feature = pd.merge(feature, user_id, on=KEY, how='left')

tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature/train')

tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')

# =============================================================================
# drop old user
# =============================================================================
dup_tr = dup[dup.is_train==1]

dup_tr['seq'] = 1
dup_tr['seq'] = dup_tr.groupby('user_id').seq.cumsum()-1

dup_tr['seq_max'] = dup_tr.groupby('user_id').seq.transform('max')

dup_tr[['SK_ID_CURR', 'user_id', 'seq', 'seq_max', 'is_train']]


dup_tr[dup_tr.seq != dup_tr.seq_max][['SK_ID_CURR', 'user_id', 'seq', 'seq_max', 'is_train']]









a = dup_tr.drop_duplicates('user_id', keep='last')['SK_ID_CURR'].tolist()
b = dup_tr[dup_tr.duplicated('user_id', keep=False)]['SK_ID_CURR'].tolist()

c = set(b)-set(a)
len(a), len(b), len(c)


#drop_ids.to_csv('../data/drop_ids.csv', index=False)


# =============================================================================
# check
# =============================================================================

a = pd.read_feather('../feature/train_f025_last_TARGET.f')['f025_last_TARGET'].to_frame()
a[tmp]

#==============================================================================
#utils.end(__file__)




