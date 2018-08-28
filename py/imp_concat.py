#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:00:33 2018

@author: kazuki.onodera
"""

import pandas as pd

li = ['/Users/kazuki.onodera/Downloads/imp_191_imp.py.csv',
      '/Users/kazuki.onodera/Downloads/imp_192_imp.py.csv',
      '/Users/kazuki.onodera/Downloads/imp_193_imp.py.csv',
      '/Users/kazuki.onodera/Downloads/imp_194_imp.py.csv',
      '/Users/kazuki.onodera/Downloads/imp_195_imp.py.csv',
      '/Users/kazuki.onodera/Downloads/imp_798_imp.py.csv',
      ]


tmp = []
for p in li:
    imp = pd.read_csv(p)
    
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    
    tmp += imp[imp['total']>0.005].feature.tolist()
#    tmp += imp[imp['split']>6].feature.tolist()

tmp = sorted(set(tmp))
df = pd.DataFrame(tmp, columns=['feature'])

df.to_csv('imp_atleast_use.csv', index=False)

