#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:36:47 2018

@author: kazuki.onodera

params:
    {
    day: 2y
    amt: abs
    delay: True
    descrete_method: quantile, 10%
    }

"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f319_'

KEY = 'SK_ID_CURR'

day_start = -365*2  # min: -2922
day_end   = -365*1  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================

df = pd.read_csv('../input/installments_payments.csv.zip')
df = df[df['DAYS_INSTALMENT'].between(day_start, day_end)].sort_values(['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_ENTRY_PAYMENT'])

df['delay'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']

df = df[df['delay']>0]
df.delay = df.delay.astype(int)

df['delay_qcut'] = pd.qcut(df['delay'], 10, duplicates='drop')
df['delay_qcut'] = df['delay_qcut'].map(lambda x: f'{int(x.left)}=<{int(x.right)}')

df['pay_qcut'] = pd.qcut(df['AMT_PAYMENT'], 10, duplicates='drop')
df['pay_qcut'] = df['pay_qcut'].map(lambda x: f'{int(x.left)}=<{int(x.right)}')

df['all1'] = 1

feature = pd.pivot_table(df, index='SK_ID_CURR', 
                         columns=['pay_qcut', 'delay_qcut'], values='all1',
                         aggfunc='sum')

feature.columns = pd.Index(['pay'+e[0] + "_delay" + e[1] for e in feature.columns.tolist()])
feature.reset_index(inplace=True)


# =============================================================================
# output
# =============================================================================
train = utils.load_train([KEY])
test = utils.load_test([KEY])

tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature/train')

tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')



#==============================================================================
utils.end(__file__)


