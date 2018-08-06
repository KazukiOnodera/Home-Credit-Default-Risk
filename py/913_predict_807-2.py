#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:02:27 2018

@author: kazuki.onodera

nohup python -u 913_predict_807-2.py 0 > LOG/log_913_predict_807-2_s0.py.txt &
nohup python -u 913_predict_807-2.py 1 > LOG/log_913_predict_807-2_s1.py.txt &
nohup python -u 913_predict_807-2.py 2 > LOG/log_913_predict_807-2_s2.py.txt &
nohup python -u 913_predict_807-2.py 3 > LOG/log_913_predict_807-2_s3.py.txt &
nohup python -u 913_predict_807-2.py 4 > LOG/log_913_predict_807-2_s4.py.txt &
nohup python -u 913_predict_807-2.py 5 > LOG/log_913_predict_807-2_s5.py.txt &
nohup python -u 913_predict_807-2.py 6 > LOG/log_913_predict_807-2_s6.py.txt &
nohup python -u 913_predict_807-2.py 7 > LOG/log_913_predict_807-2_s7.py.txt &
nohup python -u 913_predict_807-2.py 8 > LOG/log_913_predict_807-2_s8.py.txt &
nohup python -u 913_predict_807-2.py 9 > LOG/log_913_predict_807-2_s9.py.txt &

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
from collections import defaultdict
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
from glob import glob
import utils, utils_cat, utils_best
utils.start(__file__)
#==============================================================================
print(sys.argv)
SEED = int(sys.argv[1])

LOOP = 20

NROUND = 5040

SUBMIT_FILE_PATH = f'../output/807-2_{SEED}.csv.gz'

COMMENT = f'CV805_LB803 (seed87 loop100)'

EXE_SUBMIT = False

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
         
         'colsample_bytree': 0.9,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
#         'seed': SEED
         }


np.random.seed(SEED)

loader = utils_best.Loader('CV805_LB803')

# =============================================================================
# load
# =============================================================================

X_train = loader.train()
y_train = utils.read_pickles('../data/label').TARGET


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()

CAT = list( set(X_train.columns) & set(loader.category()) )

COL = X_train.columns.tolist()

X_test = loader.test()[COL]



# =============================================================================
# training
# =============================================================================
dtrain = lgb.Dataset(X_train, y_train, categorical_feature=CAT )

models = []
for i in range(LOOP):
    print(f'LOOP: {i}')
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND,
                      categorical_feature=CAT)
#    model.save_model(f'lgb{i}.model')
    models.append(model)

imp = ex.getImp(models)
imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# =============================================================================
# predict
# =============================================================================
sub = pd.read_pickle('../data/sub.p')

gc.collect()

label_name = 'TARGET'

sub[label_name] = 0
for model in models:
    y_pred = model.predict(X_test)
    sub[label_name] += pd.Series(y_pred).rank()
sub[label_name] /= LOOP
sub[label_name] /= sub[label_name].max()
sub['SK_ID_CURR'] = sub['SK_ID_CURR'].map(int)

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

print(sub[label_name].describe())


# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)

#==============================================================================
utils.end(__file__)
utils.stop_instance()




