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

def df_info(target_df, topN=10):
    
    max_row = target_df.shape[0]
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['#Nulls'] = target_df.isnull().sum()
    df['#Uniques'] = target_df.nunique()
    
    # stats
    df['Min']   = target_df.min(numeric_only=True)
    df['Mean']  = target_df.mean(numeric_only=True)
    df['Max']   = target_df.max(numeric_only=True)
    df['Std']   = target_df.std(numeric_only=True)
    
    # top 10 values
    df[f'top{topN} val'] = 0
    df[f'top{topN} cnt'] = 0
    df[f'top{topN} raito'] = 0
    for c in df.index:
        vc = target_df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        raito = list((vc.values / max_row).round(2))
        df.loc[c, f'top{topN} val'] = str(val)
        df.loc[c, f'top{topN} cnt'] = str(cnt)
        df.loc[c, f'top{topN} raito'] = str(raito)
        
    return df

def top_categories(df, category_feature, topN=30):
    return df[category_feature].value_counts().head(topN).index

def count_categories(df, category_features, topN=30, sort='freq', df2=None):
    
    for c in category_features:
        target_value = df[c].value_counts().head(topN).index
        if sort=='freq':
            order = target_value
        elif sort=='alphabetic':
            order = df[c].value_counts().head(topN).sort_index().index
        
        if df2 is not None:
            plt.subplot(1, 2, 1)
        sns.countplot(x=c, data=df[df[c].isin(order)], order=order)
        plt.xticks(rotation=90)
        
        if df2 is not None:
            plt.subplot(1, 2, 2)
            sns.countplot(x=c, data=df2[df2[c].isin(order)], order=order)
            plt.xticks(rotation=90)
        
        if df2 is not None:
            plt.suptitle(f'{c} TOP{topN}', size=25)
        else:
            plt.title(f'{c} TOP{topN}', size=25)
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

def venn_diagram(train, test, category_features, names=('train', 'test'), figsize=(18,13)):
    """
    category_features: max==6
    """
    n = int(np.ceil(len(category_features)/2))
    plt.figure(figsize=figsize)
    
    for i,c in enumerate(category_features):
        plt.subplot(int(f'{n}2{i+1}'))
        venn2([set(train[c].unique()), set(test[c].unique())], 
               set_labels = names )
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
