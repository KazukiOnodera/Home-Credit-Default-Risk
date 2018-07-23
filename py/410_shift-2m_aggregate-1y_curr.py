#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:49:25 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f410_'

KEY = 'SK_ID_CURR'

SHIFT = 2

month_start = -12*1 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# curr
# =============================================================================
col_app_money = ['app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY', 'app_AMT_GOODS_PRICE']
col_app_day = ['app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_DAYS_LAST_PHONE_CHANGE']

def get_trte():
    usecols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    usecols += ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
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

# =============================================================================
# 
# =============================================================================
df = pd.read_csv('../input/credit_card_balance.csv.zip').drop('SK_ID_PREV', axis=1)
df = df.groupby([KEY, 'MONTHS_BALANCE']).sum().reset_index()
df.sort_values([KEY, 'MONTHS_BALANCE'], inplace=True)
df.reset_index(drop=True, inplace=True)

merged = df.copy()

for i in range(1, SHIFT+1):
    df_s = df.shift(i)
    
    df_s.loc[df_s[KEY]!=df[KEY], df.columns] = 0
    
    for c in df_s.columns[2:]:
        merged[c] += df_s[c]



df = pd.merge(merged, get_trte(), on='SK_ID_CURR', how='left')

#df[col_app_day] = df[col_app_day]/30

# app
df['AMT_BALANCE-d-app_AMT_INCOME_TOTAL']    = df['AMT_BALANCE'] / df['app_AMT_INCOME_TOTAL']
df['AMT_BALANCE-d-app_AMT_CREDIT']          = df['AMT_BALANCE'] / df['app_AMT_CREDIT']
df['AMT_BALANCE-d-app_AMT_ANNUITY']         = df['AMT_BALANCE'] / df['app_AMT_ANNUITY']
df['AMT_BALANCE-d-app_AMT_GOODS_PRICE']     = df['AMT_BALANCE'] / df['app_AMT_GOODS_PRICE']

df['AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL']    = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_INCOME_TOTAL']
df['AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT']          = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_CREDIT']
df['AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY']         = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_ANNUITY']
df['AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE']     = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_GOODS_PRICE']

#for c in col_app_day:
#    print(f'MONTHS_BALANCE-m-{c}')
#    df[f'MONTHS_BALANCE-m-{c}'] = df['MONTHS_BALANCE'] - df[c]


df['AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
df['AMT_BALANCE-d-AMT_DRAWINGS_CURRENT']    = df['AMT_BALANCE'] / df['AMT_DRAWINGS_CURRENT']

df['AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL'] = df['AMT_DRAWINGS_CURRENT'] / df['AMT_CREDIT_LIMIT_ACTUAL']

df['SK_DPD-m-SK_DPD_DEF'] = df['SK_DPD'] - df['SK_DPD_DEF']
df['SK_DPD-m-SK_DPD_DEF_over0'] = (df['SK_DPD-m-SK_DPD_DEF']>0)*1
df['SK_DPD-m-SK_DPD_DEF_over5']  = (df['SK_DPD-m-SK_DPD_DEF']>5)*1
df['SK_DPD-m-SK_DPD_DEF_over10'] = (df['SK_DPD-m-SK_DPD_DEF']>10)*1
df['SK_DPD-m-SK_DPD_DEF_over15'] = (df['SK_DPD-m-SK_DPD_DEF']>15)*1
df['SK_DPD-m-SK_DPD_DEF_over20'] = (df['SK_DPD-m-SK_DPD_DEF']>20)*1
df['SK_DPD-m-SK_DPD_DEF_over25'] = (df['SK_DPD-m-SK_DPD_DEF']>25)*1

col = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
       'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
       'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
       'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD',
       'SK_DPD_DEF', 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL',
       'AMT_BALANCE-d-app_AMT_CREDIT', 'AMT_BALANCE-d-app_AMT_ANNUITY',
       'AMT_BALANCE-d-app_AMT_GOODS_PRICE', 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL',
       'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT', 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY',
       'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE', 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL',
       'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL'
       ]

df.sort_values([KEY, 'MONTHS_BALANCE'], inplace=True)
df.reset_index(drop=True, inplace=True)

def multi_cre(c):
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

pool = Pool(len(col))
callback1 = pd.concat(pool.map(multi_cre, col), axis=1)
print('===== CRE ====')
col = callback1.columns.tolist()
print(col)
pool.close()
df = pd.concat([df, callback1], axis=1)
del callback1; gc.collect()

pool = Pool(10)
callback2 = pd.concat(pool.map(multi_cre, col), axis=1)
print('===== CRE ====')
col = callback2.columns.tolist()
print(col)
pool.close()
df = pd.concat([df, callback2], axis=1)
del callback2; gc.collect()

df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
df.replace(-np.inf, np.nan, inplace=True)


cre = df
cre = cre[cre['MONTHS_BALANCE'].between(month_start, month_end)]

col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = utils.get_dummies(cre)
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby(KEY).agg({**utils_agg.cre_num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['CRE_COUNT'] = df.groupby(KEY).size()
    df_agg.reset_index(inplace=True)
    
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


