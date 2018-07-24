#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 00:21:12 2018

@author: Kazuki
"""

import pandas as pd

imp_2 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb_onlyMe.py-2 (2) 0.08.51.csv')
imp_2['split'] /= imp_2['split'].max()
imp_2['gain'] /= imp_2['gain'].max()
imp_2['total'] = imp_2['split'] + imp_2['gain']
imp_2.sort_values('total', ascending=False, inplace=True)
imp_2.set_index('feature', inplace=True)


imp_3 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb_onlyMe.py-2 (3).csv')
imp_3['split'] /= imp_3['split'].max()
imp_3['gain'] /= imp_3['gain'].max()
imp_3['total'] = imp_3['split'] + imp_3['gain']
imp_3.sort_values('total', ascending=False, inplace=True)
imp_3.set_index('feature', inplace=True)

imp = imp_2.total.rank(ascending=False).to_frame()
imp['total3'] = imp_3.total.rank(ascending=False)

imp['diff'] = abs(imp.total - imp.total3)

imp_ = imp[imp.total<=700]

imp_ = imp_[imp_.total3>700]
