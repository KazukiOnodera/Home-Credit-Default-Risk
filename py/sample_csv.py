#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 03:04:00 2018

@author: Kazuki
"""
from gc import collect
import numpy as np
import utils

def sample(n=10):
    df = utils.load_train()
#    ids = np.random.choice(df.SK_ID_CURR.unique(), size=n, replace=False)
    ids0 = np.random.choice(df[df['TARGET']==0].SK_ID_CURR.unique(), size=n, replace=False)
    ids1 = np.random.choice(df[df['TARGET']==1].SK_ID_CURR.unique(), size=n, replace=False)
    df[df.SK_ID_CURR.isin(ids0)].sort_values('SK_ID_CURR').to_csv('../sample/sample_tr_0.csv', index=False)
    df[df.SK_ID_CURR.isin(ids1)].sort_values('SK_ID_CURR').to_csv('../sample/sample_tr_1.csv', index=False)
    collect()
    
    base = utils.read_pickles('../data/POS_CASH_balance')
    df = base[base.SK_ID_CURR.isin(ids0)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('../sample/sample_POS_0.csv', index=False)
    df = base[base.SK_ID_CURR.isin(ids1)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('../sample/sample_POS_1.csv', index=False)
    collect()
    
    base = utils.read_pickles('../data/bureau')
    df = base[base.SK_ID_CURR.isin(ids0)]
    df.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], ascending=[True, False]).to_csv('../sample/sample_bure_0.csv', index=False)
    df = base[base.SK_ID_CURR.isin(ids1)]
    df.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], ascending=[True, False]).to_csv('../sample/sample_bure_1.csv', index=False)
    collect()
    
    base = utils.read_pickles('../data/credit_card_balance')
    df = base[base.SK_ID_CURR.isin(ids0)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('../sample/sample_cre_0.csv', index=False)
    df = base[base.SK_ID_CURR.isin(ids1)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('../sample/sample_cre_1.csv', index=False)
    collect()
    
    base = utils.read_pickles('../data/installments_payments')
    df = base[base.SK_ID_CURR.isin(ids0)]
    df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=[True, False]).to_csv('../sample/sample_ins_0.csv', index=False)
    df = base[base.SK_ID_CURR.isin(ids1)]
    df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=[True, False]).to_csv('../sample/sample_ins_1.csv', index=False)
    collect()
    
    base = utils.read_pickles('../data/previous_application')
    df = base[base.SK_ID_CURR.isin(ids0)]
    df.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False]).to_csv('../sample/sample_prev_0.csv', index=False)
    df = base[base.SK_ID_CURR.isin(ids1)]
    df.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False]).to_csv('../sample/sample_prev_1.csv', index=False)
    
    return

sample(100)