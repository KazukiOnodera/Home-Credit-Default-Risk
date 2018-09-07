#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 23:10:08 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
import sys
argv = sys.argv
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from sklearn.metrics import roc_auc_score
from glob import glob
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP  = 5
NFOLD = 7
HEAD  = 2000

SUBMIT_FILE_PATH = '../output/907-2.csv.gz'

EXE_SUBMIT = True
COMMENT = '2000 features'

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.7,
         'subsample': 0.9,
         'nthread': 32,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
#         'seed': SEED
         }


np.random.seed(SEED)

imp = pd.read_csv('LOG/imp_801_imp_lgb.py.csv')
imp.sort_values('total', ascending=False, inplace=True)

features = imp.head(HEAD).feature # TODO: check

#==============================================================================

def mk_submit():
    
    files_tr = ('../feature/train_' + features  + '.f').tolist()
    files_te = ('../feature/test_'  + features + '.f').tolist()
    
    # =============================================================================
    # load
    # =============================================================================
    # train
    X_train = pd.concat([
                    pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
                   ], axis=1)
    y_train = utils.read_pickles('../data/label').TARGET
    
    X_train.head().to_csv(SUBMIT_FILE_PATH.replace('.csv', '_X.csv'), 
                          index=False, compression='gzip')
    
    if X_train.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X_train.shape {X_train.shape}')
    
    gc.collect()
    
    CAT = list( set(X_train.columns) & set(utils_cat.ALL) )
    print('CAT :', CAT)
    
    COL = X_train.columns.tolist()
    
    # test
    X_test = pd.concat([
                        pd.read_feather(f) for f in tqdm(files_te, mininterval=60)
                        ], axis=1)[COL]
    
    
    # =============================================================================
    # training with cv
    # =============================================================================
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=CAT, free_raw_data=False)
    
    model_all = []
    y_pred = pd.Series(0, index=y_train.index)
    for i in range(LOOP):
        gc.collect()
        param['seed'] = i
        ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD,
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=i)
        model_all += models
        y_pred += ex.eval_oob(X_train, y_train, models, i).rank()
        
        auc_mean = roc_auc_score(y_train, y_pred)
        result = f"CV auc-mean(loop {i}): {auc_mean} {ret['auc-mean'][-1]}"
        print(result)
        utils.send_line(result)
        
    y_pred /= y_pred.max()
    
    auc_mean = roc_auc_score(y_train, y_pred)
    result = f"CV auc-mean: {auc_mean}"
    print(result)
    utils.send_line(result)
    
    # =============================================================================
    # predict
    # =============================================================================
    sub = pd.read_pickle('../data/sub.p')
    
    gc.collect()
    
    label_name = 'TARGET'
    
    sub[label_name] = 0
    for model in model_all:
        y_pred = model.predict(X_test)
        sub[label_name] += pd.Series(y_pred).rank()
    sub[label_name] /= len(model_all)
    sub[label_name] /= sub[label_name].max()
    sub['SK_ID_CURR'] = sub['SK_ID_CURR'].map(int)
    
    sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')
    
    # =============================================================================
    # submission
    # =============================================================================
    if EXE_SUBMIT:
        print('submit')
        utils.submit(SUBMIT_FILE_PATH, COMMENT)

# =============================================================================
# main
# =============================================================================

mk_submit()

#==============================================================================
utils.end(__file__)
#utils.stop_instance()



