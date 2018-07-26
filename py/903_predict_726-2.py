#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:55:32 2018

@author: kazuki.onodera

check all feature

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

SUBMIT_FILE_PATH = '../output/726-2_check_all_feature.csv.gz'

COMMENT = f'CV auc-mean(7 fold): 0.80453 + 0.00324 all(700+142)'

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

np.random.seed(SEED)
# =============================================================================
# feature
# =============================================================================

df_lb800 = pd.read_csv('../output/725-1_X.csv.gz')
df_lb804 = pd.read_csv('../output/725-2_X.csv.gz')

sub_lb800 = pd.read_csv('../output/725-1.csv.gz')
sub_lb804 = pd.read_csv('../output/725-2.csv.gz')

feature_all = sorted( set(df_lb800.columns.tolist() + df_lb804.columns.tolist()) )

file_tr = ('../feature/train_' + pd.Series(feature_all) + '.f').tolist()
file_te = ('../feature/test_'  + pd.Series(feature_all) + '.f').tolist()

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

keys = sorted([c.split('_')[0][:2] for c in df_lb800.columns])
di = defaultdict(int)
for k in keys:
    di[k] += 1
for k,v in di.items():
    print(f'{k}: {v}')

keys = sorted([c.split('_')[0][:2] for c in df_lb804.columns])
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
[50]	cv_agg's auc: 0.769605 + 0.00318649
[100]	cv_agg's auc: 0.773364 + 0.00317389
[150]	cv_agg's auc: 0.776869 + 0.00315894
[200]	cv_agg's auc: 0.780227 + 0.00317625
[250]	cv_agg's auc: 0.783251 + 0.0032431
[300]	cv_agg's auc: 0.785819 + 0.00316437
[350]	cv_agg's auc: 0.787969 + 0.00314287
[400]	cv_agg's auc: 0.789725 + 0.00310901
[450]	cv_agg's auc: 0.791197 + 0.00311429
[500]	cv_agg's auc: 0.792382 + 0.00309544
[550]	cv_agg's auc: 0.793431 + 0.00310219
[600]	cv_agg's auc: 0.794311 + 0.00310213
[650]	cv_agg's auc: 0.795067 + 0.00314616
[700]	cv_agg's auc: 0.795708 + 0.0031923
[750]	cv_agg's auc: 0.796289 + 0.0032286
[800]	cv_agg's auc: 0.796821 + 0.00320598
[850]	cv_agg's auc: 0.797318 + 0.00320012
[900]	cv_agg's auc: 0.797734 + 0.00321696
[950]	cv_agg's auc: 0.798127 + 0.00327
[1000]	cv_agg's auc: 0.798474 + 0.00328395
[1050]	cv_agg's auc: 0.798839 + 0.00329333
[1100]	cv_agg's auc: 0.799155 + 0.00329745
[1150]	cv_agg's auc: 0.799434 + 0.00328511
[1200]	cv_agg's auc: 0.799737 + 0.00325034
[1250]	cv_agg's auc: 0.799995 + 0.00326976
[1300]	cv_agg's auc: 0.800249 + 0.0032848
[1350]	cv_agg's auc: 0.800494 + 0.00329496
[1400]	cv_agg's auc: 0.800697 + 0.00334354
[1450]	cv_agg's auc: 0.800906 + 0.00333734
[1500]	cv_agg's auc: 0.801118 + 0.00334736
[1550]	cv_agg's auc: 0.801331 + 0.00335658
[1600]	cv_agg's auc: 0.801493 + 0.00333661
[1650]	cv_agg's auc: 0.801663 + 0.00334175
[1700]	cv_agg's auc: 0.801824 + 0.00331756
[1750]	cv_agg's auc: 0.801961 + 0.00330404
[1800]	cv_agg's auc: 0.802082 + 0.00329482
[1850]	cv_agg's auc: 0.802193 + 0.00330222
[1900]	cv_agg's auc: 0.802343 + 0.00330101
[1950]	cv_agg's auc: 0.802494 + 0.00326175
[2000]	cv_agg's auc: 0.802592 + 0.00328481
[2050]	cv_agg's auc: 0.802727 + 0.00327681
[2100]	cv_agg's auc: 0.802848 + 0.0033067
[2150]	cv_agg's auc: 0.802986 + 0.0033285
[2200]	cv_agg's auc: 0.803116 + 0.00333816
[2250]	cv_agg's auc: 0.803202 + 0.00335219
[2300]	cv_agg's auc: 0.803316 + 0.00335083
[2350]	cv_agg's auc: 0.803419 + 0.00335378
[2400]	cv_agg's auc: 0.803514 + 0.0033644
[2450]	cv_agg's auc: 0.803591 + 0.00336243
[2500]	cv_agg's auc: 0.803685 + 0.00334527
[2550]	cv_agg's auc: 0.803751 + 0.00334748
[2600]	cv_agg's auc: 0.803824 + 0.00334759
[2650]	cv_agg's auc: 0.803877 + 0.00336408
[2700]	cv_agg's auc: 0.80392 + 0.00334033
[2750]	cv_agg's auc: 0.803966 + 0.0033227
[2800]	cv_agg's auc: 0.804045 + 0.00332554
[2850]	cv_agg's auc: 0.804081 + 0.00332539
[2900]	cv_agg's auc: 0.804119 + 0.00330334
[2950]	cv_agg's auc: 0.804139 + 0.00331385
[3000]	cv_agg's auc: 0.804147 + 0.00330857
[3050]	cv_agg's auc: 0.804188 + 0.00329834
[3100]	cv_agg's auc: 0.804203 + 0.00329402
[3150]	cv_agg's auc: 0.804234 + 0.00328852
[3200]	cv_agg's auc: 0.80429 + 0.00329883
[3250]	cv_agg's auc: 0.804333 + 0.00330275
[3300]	cv_agg's auc: 0.804323 + 0.00329234
[3350]	cv_agg's auc: 0.804357 + 0.0032821
[3400]	cv_agg's auc: 0.80438 + 0.00328024
[3450]	cv_agg's auc: 0.80439 + 0.00325612
[3500]	cv_agg's auc: 0.804404 + 0.00324985
[3550]	cv_agg's auc: 0.804409 + 0.00326285
[3600]	cv_agg's auc: 0.804414 + 0.00326037
[3650]	cv_agg's auc: 0.804434 + 0.00323414
[3700]	cv_agg's auc: 0.804436 + 0.00325324
[3750]	cv_agg's auc: 0.804464 + 0.00325695
[3800]	cv_agg's auc: 0.804477 + 0.00326649
[3850]	cv_agg's auc: 0.804483 + 0.00324604
[3900]	cv_agg's auc: 0.804494 + 0.00324131
[3950]	cv_agg's auc: 0.80451 + 0.00323686
[4000]	cv_agg's auc: 0.804519 + 0.00323545
[4050]	cv_agg's auc: 0.804497 + 0.0032388
[4100]	cv_agg's auc: 0.804511 + 0.00325519
CV auc-mean: 0.8045302942244782 + 0.0032406113638086542

"""

NROUND = 4821

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


