#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:00:33 2018

@author: kazuki.onodera
"""

import pandas as pd

li = ['/Users/Kazuki/Downloads/imp_191_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_192_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_193_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_194_imp.py.csv',
      '/Users/Kazuki/Downloads/imp_195_imp.py.csv',
      
      ]


tmp = []
for p in li:
    imp = pd.read_csv(p)
    
    tmp += imp[imp['split']>0][imp.feature.str.startswith('f15')].feature.tolist()
    tmp += imp[imp['split']>0][imp.feature.str.startswith('f025')].feature.tolist()
    
#    imp['split'] /= imp['split'].max()
#    imp['gain'] /= imp['gain'].max()
#    imp['total'] = imp['split'] + imp['gain']
#    tmp += imp[imp['total']>0.005].feature.tolist()


#tmp += pd.read_csv('/Users/kazuki.onodera/Downloads/imp_801_imp_lgb.py-2.csv').head(1000).feature.tolist()


tmp = sorted(set(tmp))
df = pd.DataFrame(tmp, columns=['feature'])

df.to_csv('imp_atleast_use.csv', index=False)


imp1 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2-1.csv').set_index('feature')
imp2 = pd.read_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2-2.csv').set_index('feature')


tmp = (imp1.total + imp2.total).sort_values(ascending=False).to_frame().reset_index()

tmp.to_csv('/Users/Kazuki/Downloads/imp_801_imp_lgb.py-2.csv', index=False)

