#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:00:05 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

# setting
PREF = 'pos_202_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
#pos = pd.read_csv('/Users/Kazuki/Home-Credit-Default-Risk/sample/sample_POS.csv')
pos = utils.read_pickles('../data/POS_CASH_balance')
base = pos[[KEY]].drop_duplicates().set_index(KEY)

gr_c   = pos.groupby('SK_ID_CURR')
gr_cp  = pos.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
gr_cm  = pos.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])
gr_ci  = pos.groupby(['SK_ID_CURR', 'CNT_INSTALMENT'])
gr_cs  = pos.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])

# =============================================================================
# size feature
# =============================================================================
base['c_size'] = gr_c.size()

cp_size = gr_cp.size().groupby('SK_ID_CURR')
base['cp_size_min']  = cp_size.min()
base['cp_size_mean'] = cp_size.mean()
base['cp_size_max']  = cp_size.max()
base['cp_size_max-min'] = base['cp_size_max'] - base['cp_size_min']
base['p_nunique'] = cp_size.size()


cm_size = gr_cm.size().groupby('SK_ID_CURR')
base['cm_size_min']  = cm_size.min() # all1 ?
base['cm_size_mean'] = cm_size.mean()
base['cm_size_max']  = cm_size.max()
base['cm_size_max-min'] = base['cm_size_max'] - base['cm_size_min']
base['m_nunique'] = cm_size.size()


ci_size = gr_ci.size().groupby('SK_ID_CURR')
base['ci_size_min']  = ci_size.min() # all1 ?
base['ci_size_mean'] = ci_size.mean()
base['ci_size_max']  = ci_size.max()
base['ci_size_max-min'] = base['ci_size_max'] - base['ci_size_min']
base['i_nunique'] = ci_size.size()

cs_size = gr_cs.size().groupby('SK_ID_CURR')
base['cs_size_min']  = cs_size.min() # all1 ?
base['cs_size_mean'] = cs_size.mean()
base['cs_size_max']  = cs_size.max()
base['cs_size_max-min'] = base['cs_size_max'] - base['cs_size_min']
base['s_nunique'] = cs_size.size()

# =============================================================================
# NAME_CONTRACT_STATUS
# =============================================================================

ct1 = pd.crosstab(pos[KEY], pos['NAME_CONTRACT_STATUS']).add_suffix('_cnt')
ct2 = pd.crosstab(pos[KEY], pos['NAME_CONTRACT_STATUS'], normalize='index').add_suffix('_nrm')

base = pd.concat([base, ct1, ct2], axis=1)






# TODO: DPD
























# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])

test = utils.load_test([KEY])


train_ = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(train_.add_prefix(PREF), '../feature/train')

test_ = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(test_.add_prefix(PREF),  '../feature/test')


#==============================================================================
utils.end(__file__)

