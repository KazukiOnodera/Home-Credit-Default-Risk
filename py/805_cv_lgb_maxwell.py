#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 07:58:15 2018

@author: Kazuki
"""


import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
#from glob import glob
import count
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

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


categorical_feature = ['f001_NAME_CONTRACT_TYPE',
                     'f001_CODE_GENDER',
                     'f001_FLAG_OWN_CAR',
                     'f001_FLAG_OWN_REALTY',
                     'f001_NAME_TYPE_SUITE',
                     'f001_NAME_INCOME_TYPE',
                     'f001_NAME_EDUCATION_TYPE',
                     'f001_NAME_FAMILY_STATUS',
                     'f001_NAME_HOUSING_TYPE',
                     'f001_OCCUPATION_TYPE',
                     'f001_WEEKDAY_APPR_PROCESS_START',
                     'f001_ORGANIZATION_TYPE',
                     'f001_FONDKAPREMONT_MODE',
                     'f001_HOUSETYPE_MODE',
                     'f001_WALLSMATERIAL_MODE',
                     'f001_EMERGENCYSTATE_MODE']


categorical_feature += ['f108_NAME_CONTRACT_TYPE',
                     'f108_WEEKDAY_APPR_PROCESS_START',
                     'f108_NAME_CASH_LOAN_PURPOSE',
                     'f108_NAME_CONTRACT_STATUS',
                     'f108_NAME_PAYMENT_TYPE',
                     'f108_CODE_REJECT_REASON',
                     'f108_NAME_TYPE_SUITE',
                     'f108_NAME_CLIENT_TYPE',
                     'f108_NAME_GOODS_CATEGORY',
                     'f108_NAME_PORTFOLIO',
                     'f108_NAME_PRODUCT_TYPE',
                     'f108_CHANNEL_TYPE',
                     'f108_NAME_SELLER_INDUSTRY',
                     'f108_NAME_YIELD_GROUP',
                     'f108_PRODUCT_COMBINATION']

categorical_feature += ['f109_NAME_CONTRACT_TYPE',
                         'f109_WEEKDAY_APPR_PROCESS_START',
                         'f109_NAME_CASH_LOAN_PURPOSE',
                         'f109_NAME_CONTRACT_STATUS',
                         'f109_NAME_PAYMENT_TYPE',
                         'f109_CODE_REJECT_REASON',
                         'f109_NAME_TYPE_SUITE',
                         'f109_NAME_CLIENT_TYPE',
                         'f109_NAME_GOODS_CATEGORY',
                         'f109_NAME_PORTFOLIO',
                         'f109_NAME_PRODUCT_TYPE',
                         'f109_CHANNEL_TYPE',
                         'f109_NAME_SELLER_INDUSTRY',
                         'f109_NAME_YIELD_GROUP',
                         'f109_PRODUCT_COMBINATION']

categorical_feature += ['FLAG_PHONE_PATTERN', 'FLAG_DOC_PATTERN'] # MAxwell


use_files = []


# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1); gc.collect()
y = utils.read_pickles('../data/label').TARGET

maxwell = pd.read_feather('../feature_someone/Maxwell_train.f')
X = pd.concat([X, maxwell], axis=1); gc.collect()


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()


# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean: {ret['auc-mean'][-1]}"
print(result)

utils.send_line(result)


# =============================================================================
# train
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
#model = lgb.train(param, dtrain, len(ret['auc-mean']))
model = lgb.train(param, dtrain, 1000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

"""
imp = pd.read_csv('LOG/imp_909_cv.py.csv')
"""

#def multi_touch(arg):
#    os.system(f'touch "../feature_unused/{arg}.f"')
#
#
#col = imp[imp['split']==0]['feature'].tolist()
#pool = Pool(cpu_count())
#pool.map(multi_touch, col)
#pool.close()


#col = imp[imp['split']==0]['feature'].tolist()
#for c in col:
#    os.system(f'touch "../unused_feature/{c}.f"')

# =============================================================================
# 
# =============================================================================
#col = imp['index'][:20].tolist()
#dtrain = lgb.Dataset(X[col], y, categorical_feature=list( set(col)&set(categorical_feature)) )
#gc.collect()
#
#ret = lgb.cv(param, dtrain, 9999, nfold=5,
#             early_stopping_rounds=50, verbose_eval=10,
#             seed=SEED)
#
#result = f"CV auc-mean(20 features) {ret['auc-mean'][-1]}"
#print(result)
#utils.send_line(result)


#==============================================================================
utils.end(__file__)
utils.stop_instance()

