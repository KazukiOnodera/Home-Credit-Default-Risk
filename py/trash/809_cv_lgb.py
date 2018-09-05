#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 08:36:25 2018

@author: Kazuki

CV with args



cd Home-Credit-Default-Risk/py
nohup python -u 809_cv_lgb.py 1200 > LOG/log_809_cv_lgb.py.txt &

cd Home-Credit-Default-Risk/py
nohup python -u 809_cv_lgb.py 1800 > LOG/log_809_cv_lgb.py.txt &

cd Home-Credit-Default-Risk/py
nohup python -u 809_cv_lgb.py 2000 > LOG/log_809_cv_lgb.py.txt &

killall python

"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
argv = sys.argv
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

HEAD = int(argv[1])

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
# load
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py-2.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)


files = ('../feature/train_' + imp.head(HEAD).feature + '.f').tolist()
    
X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

X['nejumi'] = np.load('../feature_someone/train_nejumi.npy')

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 9999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)

result = f"CV auc-mean({SEED}:{HEAD}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)



#==============================================================================
utils.end(__file__)
utils.stop_instance()



