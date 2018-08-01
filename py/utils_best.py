#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:41:21 2018

@author: Kazuki
"""

import pandas as pd

# =============================================================================
# LB804
# =============================================================================
def load_train_LB804():
    return pd.read_feather('../data/X_train_LB0.804.f')

def load_test_LB804():
    return pd.read_feather('../data/X_test_LB0.804.f')

def load_cat_LB804():
    return ['f108_NAME_GOODS_CATEGORY', 'f002_WEEKDAY_APPR_PROCESS_START', 
            'f109_NAME_GOODS_CATEGORY', 'f108_NAME_TYPE_SUITE', 'f109_PRODUCT_COMBINATION', 
            'f002_NAME_FAMILY_STATUS', 'f002_OCCUPATION_TYPE', 'f108_PRODUCT_COMBINATION', 
            'f510_CREDIT_TYPE', 'f002_WALLSMATERIAL_MODE', 'f002_NAME_INCOME_TYPE', 
            'f002_NAME_EDUCATION_TYPE', 'f002_ORGANIZATION_TYPE', 'f509_CREDIT_TYPE']

# =============================================================================
# LB806
# =============================================================================
def load_train_LB806():
    return pd.read_pickle('../feature_someone/0727/20180727_train_rk.pkl')

def load_test_LB806():
    return pd.read_pickle('../feature_someone/0727/20180727_test_rk.pkl')

def load_cat_LB806():
    return ['FLAG_DOCUMENT_PATTERN' 'FLAG_OWN_CAR' 'FLAG_OWN_REALTY'
         'FONDKAPREMONT_MODE' 'HOUSETYPE_MODE' 'NAME_CONTRACT_TYPE'
         'NAME_EDUCATION_TYPE' 'NAME_FAMILY_STATUS' 'NAME_HOUSING_TYPE'
         'NAME_INCOME_TYPE' 'NAME_TYPE_SUITE' 'OCCUPATION_TYPE'
         'WALLSMATERIAL_MODE' 'WEEKDAY_APPR_PROCESS_START']
    
# =============================================================================
# LB806
# =============================================================================
def load_train_CV805_LB803():
    return pd.read_pickle('../data/X_train_CV805_LB803.pkl.gz')

def load_test_CV805_LB803():
    return pd.read_pickle('../data/X_test_CV805_LB803.pkl.gz')

def load_cat_CV805_LB803():
    return ['f108_PRODUCT_COMBINATION', 'f002_WALLSMATERIAL_MODE', 'f002_NAME_EDUCATION_TYPE', 
            'f109_NAME_GOODS_CATEGORY', 'f002_ORGANIZATION_TYPE', 'f108_NAME_GOODS_CATEGORY', 
            'f002_NAME_INCOME_TYPE', 'f002_NAME_FAMILY_STATUS', 'f002_WEEKDAY_APPR_PROCESS_START', 
            'f002_OCCUPATION_TYPE', 'f109_PRODUCT_COMBINATION', 'f108_NAME_TYPE_SUITE']

class Loader:
    def __init__(self, name):
        """
        LB804
        LB806
        CV805_LB803
        """
        if name not in ['LB804', 'LB806', 'CV805_LB803']:
            raise Exception(name)
        self.name = name
    
    def train(self):
        if self.name == 'LB804':
            return load_train_LB804()
        
        elif self.name == 'LB806':
            return load_train_LB806()
        
        elif self.name == 'CV805_LB803':
            return load_train_CV805_LB803()
        
        else:
            raise Exception(self.name)
        
    
    def test(self):
        if self.name == 'LB804':
            return load_test_LB804()
        
        elif self.name == 'LB806':
            return load_test_LB806()
        
        elif self.name == 'CV805_LB803':
            return load_test_CV805_LB803()
        
        else:
            raise Exception(self.name)
    
    def category(self):
        if self.name == 'LB804':
            return load_cat_LB804()
        
        elif self.name == 'LB806':
            return load_cat_LB806()
        
        elif self.name == 'CV805_LB803':
            return load_cat_CV805_LB803()
    
        else:
            raise Exception(self.name)

