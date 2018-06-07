#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:35:15 2018

@author: Kazuki
"""


import os
import pandas as pd
import gc
from multiprocessing import Pool
import multiprocessing
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'cre'

NTHREAD = multiprocessing.cpu_count()


# =============================================================================
# load
# =============================================================================
cre = utils.read_pickles('../data/credit_card_balance')
cre.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], inplace=True, ascending=[True, False])

base = cre[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# latest
# =============================================================================
latest = cre[cre['MONTHS_BALANCE']==cre.groupby('SK_ID_CURR')['MONTHS_BALANCE'].transform(max)]
c1 = 'NAME_CONTRACT_STATUS'
df_sum = pd.crosstab(latest[KEY], latest[c1])
df_sum.columns = [f'{PREF}_latest_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
df_norm = pd.crosstab(latest[KEY], latest[c1], normalize='index')
df_norm.columns = [f'{PREF}_latest_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
df = pd.concat([df_sum, df_norm], axis=1)
col = df.columns.tolist()
base = pd.concat([base, df], axis=1)
base[col] = base[col].fillna(-1)

#base[f'{PREF}_latest_CNT_INSTALMENT_min'] = latest.groupby(KEY).CNT_INSTALMENT.min()
#base[f'{PREF}_latest_CNT_INSTALMENT_mean'] = latest.groupby(KEY).CNT_INSTALMENT.mean()
#base[f'{PREF}_latest_CNT_INSTALMENT_max'] = latest.groupby(KEY).CNT_INSTALMENT.max()
#base[f'{PREF}_latest_CNT_INSTALMENT_max-min'] = base[f'{PREF}_latest_CNT_INSTALMENT_max'] - base[f'{PREF}_latest_CNT_INSTALMENT_min']



# =============================================================================
# binary features
# =============================================================================
col_binary = []
for i in range(1, 11):
    cre[f'SK_DPD_over{i}'] = (cre.SK_DPD>=i)*1
    col_binary.append(f'SK_DPD_over{i}')

for c in ['Active', 'Completed', 'Signed', 'Sent proposal', 'Refused', 'Demand', 'Approved']:
    cre[f'is_{c.replace(" ", "-")}'] = (cre.NAME_CONTRACT_STATUS==c)*1
    col_binary.append(f'is_{c.replace(" ", "-")}')

col_binary.append('AMT_CREDIT_LIMIT_ACTUAL_dec')
col_binary.append('AMT_CREDIT_LIMIT_ACTUAL_inc')


ids = cre.SK_ID_CURR.unique()
all_months = pd.DataFrame(list(range(-96, 0)), columns=['MONTHS_BALANCE'])

def to_decimal(x):
    x = ''.join(map(str, x))[::-1]
    return float(x[0] + '.' + x[1:])

def multi(id_curr):
    tmp = cre[cre.SK_ID_CURR==id_curr]
    tmp['AMT_CREDIT_LIMIT_ACTUAL'] = tmp['AMT_CREDIT_LIMIT_ACTUAL'].diff(-1)
    tmp['AMT_CREDIT_LIMIT_ACTUAL_dec'] = (tmp['AMT_CREDIT_LIMIT_ACTUAL']<0)*1
    tmp['AMT_CREDIT_LIMIT_ACTUAL_inc'] = (tmp['AMT_CREDIT_LIMIT_ACTUAL']>0)*1
    shortage = all_months[~all_months.MONTHS_BALANCE.isin(tmp['MONTHS_BALANCE'])]
    shortage['SK_ID_CURR'] = id_curr
    tmp2 = pd.concat([shortage, tmp]).sort_values(['MONTHS_BALANCE'], ascending=False).fillna(0)
    tmp2[col_binary] = tmp2[col_binary].astype(int)
    gr = tmp2.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])
    
    tmp_min = gr[col_binary].min().apply(to_decimal)
    tmp_max = gr[col_binary].max().apply(to_decimal)
    tmp_diff = tmp_max = tmp_min
    tmp = pd.concat([
                     tmp_min.add_prefix(f'{PREF}_').add_suffix('_min-ts'), 
                     tmp_max.add_prefix(f'{PREF}_').add_suffix('_max-ts'),
                     tmp_diff.add_prefix(f'{PREF}_').add_suffix('_max-min-ts')
                     ])
    tmp['SK_ID_CURR'] = id_curr
    return tmp

# =============================================================================
# main
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi, ids)
pool.close()

df = pd.concat(callback, axis=1).T.set_index('SK_ID_CURR')
base = pd.concat([base, df], axis=1)

# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)
if base.columns.duplicated().sum() != 0:
    raise Exception( base.columns[base.columns.duplicated()] )

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(train, '../data/402_train', utils.SPLIT_SIZE)
del train; gc.collect()


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(test,  '../data/402_test',  utils.SPLIT_SIZE)
del test; gc.collect()




#==============================================================================
utils.end(__file__)

