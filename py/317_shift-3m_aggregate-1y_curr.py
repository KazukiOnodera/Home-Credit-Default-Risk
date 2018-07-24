#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 19:14:20 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc, os
from multiprocessing import Pool
#NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f317_'

KEY = 'SK_ID_CURR'

SHIFT = 3

month_start = -12*1 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================

df = pd.read_csv('../input/installments_payments.csv.zip',
                 usecols=['SK_ID_CURR', 'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT',
                          'AMT_INSTALMENT', 'AMT_PAYMENT'])


df['days_delayed_payment'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
df['amt_ratio'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
df['amt_delta'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
df['days_weighted_delay'] = df['amt_ratio'] * df['days_delayed_payment']
df['month'] = (df['DAYS_ENTRY_PAYMENT']/30).map(np.floor)


df = df.groupby([KEY, 'month']).sum().reset_index()
df.drop(['DAYS_ENTRY_PAYMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT'], 
        axis=1, inplace=True)

df.sort_values([KEY, 'month'], inplace=True)
df.reset_index(drop=True, inplace=True)

merged = df.copy()

for i in range(1, SHIFT+1):
    df_s = df.shift(i)
    
    df_s.loc[df_s[KEY]!=df[KEY], df.columns] = 0
    
    for c in df_s.columns[2:]:
        merged[c] += df_s[c]


col_app_money = ['app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY', 'app_AMT_GOODS_PRICE']
col_app_day = ['app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_DAYS_LAST_PHONE_CHANGE']

def get_trte():
    usecols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
#    usecols += ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    rename_di = {
                 'AMT_INCOME_TOTAL':       'app_AMT_INCOME_TOTAL', 
                 'AMT_CREDIT':             'app_AMT_CREDIT', 
                 'AMT_ANNUITY':            'app_AMT_ANNUITY',
                 'AMT_GOODS_PRICE':        'app_AMT_GOODS_PRICE',
                 'DAYS_BIRTH':             'app_DAYS_BIRTH', 
                 'DAYS_EMPLOYED':          'app_DAYS_EMPLOYED', 
                 'DAYS_REGISTRATION':      'app_DAYS_REGISTRATION', 
                 'DAYS_ID_PUBLISH':        'app_DAYS_ID_PUBLISH', 
                 'DAYS_LAST_PHONE_CHANGE': 'app_DAYS_LAST_PHONE_CHANGE',
                 }
    trte = pd.concat([pd.read_csv('../input/application_train.csv.zip', usecols=usecols).rename(columns=rename_di), 
                      pd.read_csv('../input/application_test.csv.zip',  usecols=usecols).rename(columns=rename_di)],
                      ignore_index=True)
    return trte

df = pd.merge(merged, get_trte(), on=KEY, how='left')


# app
df['AMT_PAYMENT-d-app_AMT_INCOME_TOTAL'] = df['AMT_PAYMENT'] / df['app_AMT_INCOME_TOTAL']
df['AMT_PAYMENT-d-app_AMT_CREDIT']      = df['AMT_PAYMENT'] / df['app_AMT_CREDIT']
df['AMT_PAYMENT-d-app_AMT_ANNUITY']     = df['AMT_PAYMENT'] / df['app_AMT_ANNUITY']
df['AMT_PAYMENT-d-app_AMT_GOODS_PRICE'] = df['AMT_PAYMENT'] / df['app_AMT_GOODS_PRICE']

# prev
df['AMT_PAYMENT-d-AMT_ANNUITY'] = df['AMT_PAYMENT'] / df['AMT_ANNUITY']
df['AMT_PAYMENT-m-AMT_ANNUITY'] = df['AMT_PAYMENT'] - df['AMT_ANNUITY']

df.sort_values([KEY, 'month'], inplace=True)
df.reset_index(drop=True, inplace=True)

def multi_ins(c):
    ret_diff = []
    ret_pctchng = []
    key_bk = x_bk = None
    for key, x in df[[KEY, c]].values:
        
        if key_bk is None:
            ret_diff.append(None)
            ret_pctchng.append(None)
        else:
            if key_bk == key:
                ret_diff.append(x - x_bk)
                ret_pctchng.append( (x_bk-x) / x_bk)
            else:
                ret_diff.append(None)
                ret_pctchng.append(None)
        key_bk = key
        x_bk = x
        
    ret_diff = pd.Series(ret_diff, name=f'{c}_diff')
    ret_pctchng = pd.Series(ret_pctchng, name=f'{c}_pctchange')
    ret = pd.concat([ret_diff, ret_pctchng], axis=1)
    
    return ret

col = ['days_delayed_payment', 'amt_ratio', 'amt_delta', 'days_weighted_delay',
       'AMT_PAYMENT',
       'AMT_PAYMENT-d-app_AMT_INCOME_TOTAL', 'AMT_PAYMENT-d-app_AMT_CREDIT',
       'AMT_PAYMENT-d-app_AMT_ANNUITY', 'AMT_PAYMENT-d-app_AMT_GOODS_PRICE',
       'AMT_PAYMENT-d-AMT_ANNUITY', 'AMT_PAYMENT-m-AMT_ANNUITY']
pool = Pool(len(col))
callback1 = pd.concat(pool.map(multi_ins, col), axis=1)
print('===== INS ====')
col = callback1.columns.tolist()
print(col)
pool.close()
df = pd.concat([df, callback1], axis=1)
del callback1; gc.collect()

#pool = Pool(3)
#callback2 = pd.concat(pool.map(multi_ins, col), axis=1)
#print('===== INS ====')
#col = callback2.columns.tolist()
#print(col)
#pool.close()
#df = pd.concat([df, callback2], axis=1)
#del callback2; gc.collect()

df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
df.replace(-np.inf, np.nan, inplace=True)


df = df[df['month'].between(month_start, month_end)]

train = utils.load_train([KEY])
test = utils.load_test([KEY])


df.drop(['app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY',
       'app_AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_PAYMENT'], axis=1, inplace=True)

stats = ['min', 'mean', 'max', 'var', 'sum']
num_aggregations = {}
for c in df.columns[2:]:
    num_aggregations[c] = stats

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df_agg = df.groupby(KEY).agg({**num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg.reset_index(inplace=True)
    
#    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

aggregate()



#==============================================================================
utils.end(__file__)


