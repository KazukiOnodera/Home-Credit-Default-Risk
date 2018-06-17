#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:18:55 2018

@author: Kazuki
"""

from glob import glob
import sys
argv = sys.argv

files = glob(f'../feature/{argv[1]}*.f')

print(len(files))

