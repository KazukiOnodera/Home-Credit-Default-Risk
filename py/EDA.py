#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:28:58 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

def df_info(target_df):
    
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['#Nulls'] = target_df.isnull().sum()
    df['#Uniques'] = target_df.nunique()
    
    return df

def top_categories(df, category_feature, top=30):
    return df[category_feature].value_counts()[:top].index

def count_categories(df, category_features, top=30, sort='freq', df2=None):
    
    for c in category_features:
        target_value = df[c].value_counts()[:top].index
        if sort=='freq':
            order = target_value
        elif sort=='alphabetic':
            order = df[c].value_counts()[:top].sort_index().index
        
        if df2 is not None:
            plt.subplot(1, 2, 1)
        sns.countplot(x=c, data=df[df[c].isin(order)], order=order)
        plt.xticks(rotation=90)
        
        if df2 is not None:
            plt.subplot(1, 2, 2)
            sns.countplot(x=c, data=df2[df2[c].isin(order)], order=order)
            plt.xticks(rotation=90)
        
        if df2 is not None:
            plt.suptitle(f'{c} TOP{top}', size=25)
        else:
            plt.title(f'{c} TOP{top}', size=25)
        plt.tight_layout()
        plt.show()
        
    return

def hist_continuous(df, continuous_features, bins=30, df2=None):
    
    for c in continuous_features:
        if df2 is not None:
            plt.subplot(1, 2, 1)
        df[c].hist(bins=bins)
        
        if df2 is not None:
            plt.subplot(1, 2, 2)
            df2[c].hist(bins=bins)
            
        if df2 is not None:
            plt.suptitle(f'{c}', size=25)
        else:
            plt.title(f'{c}', size=25)
        plt.tight_layout()
        plt.show()
        
    return

def venn_diagram(train, test, category_features, figsize=(18,13)):
    """
    category_features: max==6
    """
    n = int(np.ceil(len(category_features)/2))
    plt.figure(figsize=figsize)
    
    for i,c in enumerate(category_features):
        plt.subplot(int(f'{n}2{i+1}'))
        venn2([set(train[c].unique()), set(test[c].unique())], 
               set_labels = ('train', 'test') )
        plt.title(f'{c}', fontsize=18)
    plt.show()
    
    return

def split_seq(iterable, size):
    """
    In: list(split_seq(range(9), 4))
    Out: [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    """
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))
