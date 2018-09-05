#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:25:20 2018

@author: kazuki.onodera

cd Home-Credit-Default-Risk/py
python run.py 817_cv_LB804_Branden.py

"""


import gc, os
#from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
#from glob import glob
#import count
import utils, utils_best
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(99999)

NFOLD = 7

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


loader = utils_best.Loader('LB804')

category_branden = ['Bra_papp_max_SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_int',
 'Bra_papp_min_SK_ID_CURR_PRODUCT_COMBINATION_int',
 'Bra_WEEKDAY_APPR_PROCESS_START_int',
 'Bra_papp_max_SK_ID_CURR_NAME_TYPE_SUITE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_GOODS_CATEGORY_int',
 'Bra_papp_min_SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_int',
 'Bra_papp_min_SK_ID_CURR_NAME_SELLER_INDUSTRY_int',
 'Bra_CODE_GENDER_int',
 'Bra_papp_min_SK_ID_CURR_CODE_REJECT_REASON_int',
 'Bra_papp_max_SK_ID_CURR_PRODUCT_COMBINATION_int',
 'Bra_papp_min_SK_ID_CURR_NAME_CONTRACT_STATUS_int',
 'Bra_NAME_FAMILY_STATUS_int',
 'Bra_papp_max_SK_ID_CURR_NAME_GOODS_CATEGORY_int',
 'Bra_OCCUPATION_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_SELLER_INDUSTRY_int',
 'Bra_papp_max_SK_ID_CURR_CODE_REJECT_REASON_int',
 'Bra_papp_min_SK_ID_CURR_NAME_YIELD_GROUP_int',
 'Bra_papp_min_SK_ID_CURR_NAME_PRODUCT_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_CHANNEL_TYPE_int',
 'Bra_WALLSMATERIAL_MODE_int',
 'Bra_ORGANIZATION_TYPE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_CLIENT_TYPE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_CONTRACT_STATUS_int',
 'Bra_papp_min_SK_ID_CURR_NAME_PORTFOLIO_int',
 'Bra_papp_max_SK_ID_CURR_NAME_YIELD_GROUP_int',
 'Bra_papp_min_SK_ID_CURR_CHANNEL_TYPE_int',
 'Bra_NAME_EDUCATION_TYPE_int',
 'Bra_FONDKAPREMONT_MODE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_PRODUCT_TYPE_int',
 'Bra_NAME_INCOME_TYPE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_CONTRACT_TYPE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_TYPE_SUITE_int',
 'Bra_NAME_TYPE_SUITE_int',
 'Bra_NAME_HOUSING_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_PAYMENT_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_CLIENT_TYPE_int',
 'Bra_papp_min_SK_ID_CURR_NAME_PAYMENT_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_CONTRACT_TYPE_int',
 'Bra_papp_max_SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_int']

print('seed:', SEED)

# =============================================================================
# load
# =============================================================================
X = pd.concat([
        loader.train(),
        pd.read_feather('../feature_someone/branden/X_train.f')
        ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(loader.category()) ) + category_branden

print('category:', CAT)
# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)

result = f"CV auc-mean(seed:{SEED}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)
imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# =============================================================================
# predict
# =============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X_test = pd.concat([
        loader.test(),
        pd.read_feather('../feature_someone/branden/X_test.f')
        ], axis=1)[X.columns]

sub_train = pd.DataFrame(index=X.index)
sub_test  = pd.DataFrame(index=X_test.index)

sub_train['y'] = y
sub_train['y_pred'] = 0
sub_test['y_pred'] = 0

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
for (train_index, valid_index),model in zip(skf.split(X, y), models):
    X_val, y_val = X.iloc[valid_index], y.iloc[valid_index]
    y_pred = model.predict(X_val)
    print(roc_auc_score(y_val.values, y_pred))
    
    sub_train.iloc[valid_index, -1] = y_pred
    sub_test['y_pred'] += model.predict(X_test)

sub_test['y_pred'] /= NFOLD

# =============================================================================
# save
# =============================================================================
sub_train.to_csv(f'../data/LB804_Branden_train_pred_s{SEED}.csv', index=False)
sub_test.to_csv(f'../data/LB804_Branden_test_pred_s{SEED}.csv', index=False)


#==============================================================================
utils.end(__file__)
utils.stop_instance()



