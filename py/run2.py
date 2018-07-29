#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 01:51:19 2018

@author: Kazuki
"""


import os
from time import sleep
import sys
argv = sys.argv

file1 = argv[1]
file2 = argv[2]


os.system(f'python -u {file1} > LOG/log_{file1}.txt &')

os.system(f'python -u {file2} > LOG/log_{file2}.txt &')

