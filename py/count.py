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
    files = glob(f'../feature/{argv[1]}*.f')
    print(len(files))
    
else:
    files = [f.split('/')[-1] for f in sorted(glob(f'../feature/*.f'))]
    
    print('train files')
    keys = sorted([f.split('_')[1] for f in files if f.startswith('train_')])
    di = defaultdict(int)
    for k in keys:
        di[k] += 1
    for k,v in di.items():
        print(f'{k}: {v}')
    
    print('\ntest files')
    keys = sorted([f.split('_')[1] for f in files if f.startswith('test_')])
    di = defaultdict(int)
    for k in keys:
        di[k] += 1
    for k,v in di.items():
        print(f'{k}: {v}')
    
