#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:22:58 2018

@author: Kazuki
"""

from glob import glob
import os

files = sorted(glob('../feature/train*.f'))
for f in files:
    path = f.replace('train_', 'test_', 1)
    if not os.path.isfile(path):
        print(f)



files = sorted(glob('../feature/test*.f'))
for f in files:
    path = f.replace('test_', 'train_', 1)
    if not os.path.isfile(path):
        print(f)


