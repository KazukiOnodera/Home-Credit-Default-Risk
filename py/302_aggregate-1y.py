#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 00:45:09 2018

@author: Kazuki

based on
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
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
PREF = 'f302_'

KEY = 'SK_ID_CURR'

day_start = -365*1  # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
ins = utils.read_pickles('../data/installments_payments')
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]

ins_d = utils.read_pickles('../data/installments_payments_delay')
ins_d = ins_d[ins_d['DAYS_INSTALMENT'].between(day_start, day_end)]
ins_d.columns = ins_d.columns[:2].tolist() + ('delay_' + ins_d.columns[2:]).tolist()

ins_nd = utils.read_pickles('../data/installments_payments_notdelay')
ins_nd = ins_nd[ins_nd['DAYS_INSTALMENT'].between(day_start, day_end)]
ins_nd.columns = ins_nd.columns[:2].tolist() + ('notdelay_' + ins_nd.columns[2:]).tolist()


#col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate(df, pref):
    
    df_agg = df.groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
    
#    gr1 = df.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER'])
#    df_agg = gr1.sum().groupby('SK_ID_CURR').agg({**utils_agg.ins_num_aggregations})
    
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg[pref + 'INS_COUNT'] = df.groupby('SK_ID_CURR').size()
    df_agg.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

aggregate(ins,    '')
aggregate(ins_d,  'delay_')
aggregate(ins_nd, 'notdelay_')






#==============================================================================
utils.end(__file__)
