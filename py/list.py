#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 01:57:53 2018

@author: Kazuki
"""

from glob import glob

files = sorted(glob('*.py'))

for fi in files:
    print(f'python run.py {fi.split("/")[-1]}')
