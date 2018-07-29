#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:18:55 2018

@author: Kazuki
"""

from glob import glob
from collections import defaultdict
import sys
argv = sys.argv

if len(argv)>1:
    files = glob(f'../feature_bureau/{argv[1]}*.f')
    print(len(files))
    
else:
    files = glob(f'../feature_bureau/*.f')
    
    print('train files')
    keys = sorted([f.split('/')[-1].split('_')[1] for f in files if 'train_' in f])
    di = defaultdict(int)
    for k in keys:
        di[k] += 1
    for k,v in di.items():
        print(f'{k}: {v}')
    
    print('\ntest files')
    keys = sorted([f.split('/')[-1].split('_')[1] for f in files if 'test_' in f])
    di = defaultdict(int)
    for k in keys:
        di[k] += 1
    for k,v in di.items():
        print(f'{k}: {v}')
    
