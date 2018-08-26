#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:23:05 2018

@author: Kazuki
"""


import pandas as pd

# =============================================================================

version = 3

date_feature = [
                'DAYS_BIRTH',
                'DAYS_REGISTRATION',
                'DAYS_ID_PUBLISH',
                'DAYS_EMPLOYED'
                ]

other_feature = [
                'CODE_GENDER',
                'NAME_EDUCATION_TYPE',
#                'NAME_FAMILY_STATUS',
#                'NAME_HOUSING_TYPE'
                ]

# =============================================================================

train  = pd.read_csv('../input/application_train.csv.zip')
train['is_train'] = 1

test  = pd.read_csv('../input/application_test.csv.zip')
test['is_train'] = 0

trte = pd.concat([train, test], ignore_index=True)


# new key
trte['DAYS_BIRTH-m-DAYS_REGISTRATION'] = trte['DAYS_BIRTH'] - trte['DAYS_REGISTRATION']
trte['DAYS_REGISTRATION-m-DAYS_ID_PUBLISH']   = trte['DAYS_REGISTRATION'] - trte['DAYS_ID_PUBLISH']
trte['DAYS_ID_PUBLISH-m-DAYS_EMPLOYED']   = trte['DAYS_ID_PUBLISH'] - trte['DAYS_EMPLOYED']

keys = ['DAYS_BIRTH-m-DAYS_REGISTRATION', 'DAYS_REGISTRATION-m-DAYS_ID_PUBLISH',
         'DAYS_ID_PUBLISH-m-DAYS_EMPLOYED'] + other_feature


tmp = trte[trte.duplicated(keys, False)].sort_values(keys)

cnt = tmp.groupby(keys).size()
cnt.name = 'dup_cnt'
cnt = cnt.reset_index()

cnt['dup_id'] = range(cnt.shape[0])

dup_tbl = pd.merge(tmp, cnt, on=keys, how='left')
col = train.columns.tolist() + [c for c in dup_tbl.columns if c not in train.columns]

dup_tbl.sort_values(['dup_id', 'DAYS_BIRTH'], inplace=True)

dup_tbl[col].to_csv(f'../data/same_user_all_v{version}.csv.gz', index=False, compression='gzip')



# =============================================================================
# user id
# =============================================================================
user = trte[['SK_ID_CURR']].sort_values('SK_ID_CURR')
user_dup   = user[user.SK_ID_CURR.isin(dup_tbl.SK_ID_CURR)]
user_other = user[~user.SK_ID_CURR.isin(dup_tbl.SK_ID_CURR)]

user_dup = pd.merge(user_dup, dup_tbl[['SK_ID_CURR', 'dup_id']], on='SK_ID_CURR', how='left')
user_dup.rename(columns={'dup_id': 'user_id'}, inplace=True)

start = user_dup.user_id.max() +1
user_other['user_id'] = range(start, start+user_other.shape[0])


user = pd.concat([user_dup, user_other]).sort_values('SK_ID_CURR').reset_index(drop=True)
user.to_csv(f'../data/user_id_v{version}.csv.gz', index=False, compression='gzip')


