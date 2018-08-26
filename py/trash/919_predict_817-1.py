#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:11:32 2018

@author: Kazuki
"""

import pandas as pd
from glob import glob
import utils
from sklearn.metrics import roc_auc_score

SUBMIT_FILE_PATH = '../output/817-1_LB804+Branden-7725.csv.gz'


# =============================================================================
# train
# =============================================================================
files = sorted( glob('../data/LB804_Branden_train_pred_*') )

sub = utils.read_pickles('../data/label')
sub['pred'] = 0

for f in (files):
    sub['pred'] += pd.read_csv(f).y_pred.rank().values
sub['pred'] /= len(files)
sub['pred'] /= sub['pred'].max()


auc_mean = roc_auc_score(sub['TARGET'], sub['pred'])


COMMENT = f'CV(7fold*4loop, rank avg): {auc_mean}'



# =============================================================================
# test
# =============================================================================

files = sorted( glob('../data/LB804_Branden_test_pred_*') )

sub = pd.read_pickle('../data/sub.p')
sub['TARGET'] = 0

for f in (files):
    sub['TARGET'] += pd.read_csv(f).y_pred.rank().values
    
sub['TARGET'] /= len(files)

sub['TARGET'] /= sub['TARGET'].max()

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
print('submit')
utils.submit(SUBMIT_FILE_PATH, COMMENT)

# =============================================================================
# for blending
# =============================================================================
files = sorted( glob('../data/LB804_Branden_train_pred_*') )

train = utils.read_pickles('../data/label')
train['TARGET'] = 0

for f in (files):
    train['TARGET'] += pd.read_csv(f).y_pred.rank().values
train['TARGET'] /= len(files)
train['TARGET'] /= train['TARGET'].max()

train.to_csv('../output/817-1_LB804+Branden-7725_cv.csv.gz', index=False,
             compression='gzip')





