#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 03:04:00 2018

@author: Kazuki
"""
from glob import glob
import numpy as np
import utils

def sample(n=10):
    df = utils.load_train()
    ids = np.random.choice(df.SK_ID_CURR.unique(), size=n, replace=False)
    df[df.SK_ID_CURR.isin(ids)].sort_values('SK_ID_CURR').to_csv('sample_tr.csv', index=False)
    
    df = utils.read_pickles('../data/POS_CASH_balance')
    df = df[df.SK_ID_CURR.isin(ids)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('sample_POS.csv', index=False)
    
    df = utils.read_pickles('../data/bureau')
    df = df[df.SK_ID_CURR.isin(ids)]
    df.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], ascending=[True, False]).to_csv('sample_bure.csv', index=False)
    
    df = utils.read_pickles('../data/credit_card_balance')
    df = df[df.SK_ID_CURR.isin(ids)]
    df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False]).to_csv('sample_cre.csv', index=False)
    
    df = utils.read_pickles('../data/installments_payments')
    df = df[df.SK_ID_CURR.isin(ids)]
    df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=[True, False]).to_csv('sample_ins.csv', index=False)
    
    df = utils.read_pickles('../data/previous_application')
    df = df[df.SK_ID_CURR.isin(ids)]
    df.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False]).to_csv('sample_prev.csv', index=False)
    
    return

sample(100)