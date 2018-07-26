#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:44:20 2018

@author: kazuki.onodera

check same feature

"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from glob import glob
from collections import defaultdict
import utils, utils_cat
utils.start(__file__)
#==============================================================================

LOOP = 3

SEED = 71

SUBMIT_FILE_PATH = '../output/726-1_check_same_feature.csv.gz'

COMMENT = f'CV auc-mean(7 fold): 0.80365 + 0.00365 all(458)+nejumi'

EXE_SUBMIT = True

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
         'seed': SEED
         }

# =============================================================================
# feature
# =============================================================================

df_lb800 = pd.read_csv('../output/725-1_X.csv.gz')
df_lb804 = pd.read_csv('../output/725-2_X.csv.gz')

sub_lb800 = pd.read_csv('../output/725-1.csv.gz')
sub_lb804 = pd.read_csv('../output/725-2.csv.gz')

feature_same = sorted(list( set(df_lb800.columns) &  set(df_lb804.columns) ))

file_tr = ('../feature/train_' + pd.Series(feature_same) + '.f').tolist()
file_te = ('../feature/test_'  + pd.Series(feature_same) + '.f').tolist()

# =============================================================================
# load data
# =============================================================================
X_train = pd.concat([
                    pd.read_feather(f) for f in tqdm(file_tr, mininterval=60)
                   ], axis=1)

X_test = pd.concat([
                    pd.read_feather(f) for f in tqdm(file_te, mininterval=60)
                   ], axis=1)[X_train.columns]

y = utils.read_pickles('../data/label').TARGET

if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[ X_train.columns.duplicated() ] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()

CAT = list( set(X_train.columns)&set(utils_cat.ALL))

print(f'category: {CAT}')

keys = sorted([c.split('_')[0] for c in X_train.columns])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')

# =============================================================================
# cv
# =============================================================================
#dtrain = lgb.Dataset(X_train, y, categorical_feature=CAT )
#gc.collect()
#
#ret = lgb.cv(param, dtrain, 9999, nfold=7,
#             early_stopping_rounds=100, verbose_eval=50,
#             seed=SEED)
#
#result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
#print(result)
#
#utils.send_line(result)
#
#NROUND = int(len(ret['auc-mean']) * 1.2)
"""
[50]	cv_agg's auc: 0.769441 + 0.00325767
[100]	cv_agg's auc: 0.773261 + 0.00320789
[150]	cv_agg's auc: 0.776716 + 0.00319956
[200]	cv_agg's auc: 0.779987 + 0.00331283
[250]	cv_agg's auc: 0.783071 + 0.00323869
[300]	cv_agg's auc: 0.785628 + 0.00315808
[350]	cv_agg's auc: 0.787761 + 0.00319356
[400]	cv_agg's auc: 0.789473 + 0.00311491
[450]	cv_agg's auc: 0.790876 + 0.00314561
[500]	cv_agg's auc: 0.792006 + 0.00313943
[550]	cv_agg's auc: 0.793008 + 0.00318709
[600]	cv_agg's auc: 0.793915 + 0.00320531
[650]	cv_agg's auc: 0.79467 + 0.00319171
[700]	cv_agg's auc: 0.795291 + 0.00319657
[750]	cv_agg's auc: 0.795863 + 0.00325596
[800]	cv_agg's auc: 0.79638 + 0.00327567
[850]	cv_agg's auc: 0.796851 + 0.00330852
[900]	cv_agg's auc: 0.797296 + 0.00331106
[950]	cv_agg's auc: 0.797639 + 0.00338574
[1000]	cv_agg's auc: 0.798031 + 0.00338775
[1050]	cv_agg's auc: 0.798397 + 0.00341045
[1100]	cv_agg's auc: 0.798724 + 0.00340354
[1150]	cv_agg's auc: 0.79902 + 0.00344836
[1200]	cv_agg's auc: 0.799323 + 0.00344263
[1250]	cv_agg's auc: 0.799562 + 0.00345141
[1300]	cv_agg's auc: 0.799796 + 0.00344767
[1350]	cv_agg's auc: 0.800036 + 0.00345908
[1400]	cv_agg's auc: 0.800263 + 0.00346711
[1450]	cv_agg's auc: 0.800466 + 0.0034373
[1500]	cv_agg's auc: 0.800667 + 0.00346906
[1550]	cv_agg's auc: 0.800875 + 0.00348931
[1600]	cv_agg's auc: 0.801041 + 0.003516
[1650]	cv_agg's auc: 0.801186 + 0.00356425
[1700]	cv_agg's auc: 0.801373 + 0.00355287
[1750]	cv_agg's auc: 0.801553 + 0.00355489
[1800]	cv_agg's auc: 0.801682 + 0.00353879
[1850]	cv_agg's auc: 0.801797 + 0.00354717
[1900]	cv_agg's auc: 0.801916 + 0.00361605
[1950]	cv_agg's auc: 0.802021 + 0.00363786
[2000]	cv_agg's auc: 0.80211 + 0.00365039
[2050]	cv_agg's auc: 0.802224 + 0.00364696
[2100]	cv_agg's auc: 0.802327 + 0.00364985
[2150]	cv_agg's auc: 0.802471 + 0.00363302
[2200]	cv_agg's auc: 0.802564 + 0.00363749
[2250]	cv_agg's auc: 0.80266 + 0.00366303
[2300]	cv_agg's auc: 0.80275 + 0.00364577
[2350]	cv_agg's auc: 0.802821 + 0.00364797
[2400]	cv_agg's auc: 0.802894 + 0.00366381
[2450]	cv_agg's auc: 0.802961 + 0.0036624
[2500]	cv_agg's auc: 0.803018 + 0.00367512
[2550]	cv_agg's auc: 0.803071 + 0.00365462
[2600]	cv_agg's auc: 0.803142 + 0.00368243
[2650]	cv_agg's auc: 0.803207 + 0.00368258
[2700]	cv_agg's auc: 0.803249 + 0.00367345
[2750]	cv_agg's auc: 0.803299 + 0.00366559
[2800]	cv_agg's auc: 0.803309 + 0.00364872
[2850]	cv_agg's auc: 0.803348 + 0.00365506
[2900]	cv_agg's auc: 0.803398 + 0.00363604
[2950]	cv_agg's auc: 0.803419 + 0.00363725
[3000]	cv_agg's auc: 0.803448 + 0.00366605
[3050]	cv_agg's auc: 0.803487 + 0.00367054
[3100]	cv_agg's auc: 0.803511 + 0.00365228
[3150]	cv_agg's auc: 0.803528 + 0.00364872
[3200]	cv_agg's auc: 0.803555 + 0.00363384
[3250]	cv_agg's auc: 0.803574 + 0.00364759
[3300]	cv_agg's auc: 0.803573 + 0.00364122
[3350]	cv_agg's auc: 0.803596 + 0.00366336
[3400]	cv_agg's auc: 0.803603 + 0.00366222
[3450]	cv_agg's auc: 0.80361 + 0.00367428
[3500]	cv_agg's auc: 0.803622 + 0.00366045
[3550]	cv_agg's auc: 0.803646 + 0.00366033
[3600]	cv_agg's auc: 0.803635 + 0.00365899
[3650]	cv_agg's auc: 0.803644 + 0.00365285
CV auc-mean: 0.8036550020096204 + 0.003659788601717347
"""

NROUND = 4267

# =============================================================================
# training
# =============================================================================
dtrain = lgb.Dataset(X_train, y, categorical_feature=CAT )

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

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)

#==============================================================================
utils.end(__file__)


