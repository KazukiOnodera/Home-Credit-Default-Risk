#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:56:34 2018

@author: kazuki.onodera
"""
import numpy as np
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================

bure = utils.read_pickles('../data/bureau')

types = bure['CREDIT_TYPE'].unique()
bure[bure['CREDIT_TYPE']==]



#==============================================================================
utils.end(__file__)

