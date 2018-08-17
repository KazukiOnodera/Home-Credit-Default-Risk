#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:21:28 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd

files = glob('/Users/Kazuki/Downloads/imp_815_imp_lgb_loop.py-s*.csv')

files = glob('/Users/kazuki.onodera/Downloads/imp_815_imp_lgb_loop.py-s*.csv')

for i,f in enumerate(files):
    imp_ = pd.read_csv(f, index_col='feature')
    if i==0:
        imp = imp_
        imp.total = imp.total.rank(ascending=False)
    else:
        imp.total += imp_.total.rank(ascending=False)

imp.sort_values('total', inplace=True)
#imp.reset_index(inplace=True)

#imp.to_csv('imp_815_imp_lgb_loop.py_without_nejumi.csv')
imp.to_csv('imp_815_imp_lgb_loop.py_Tam.csv')


