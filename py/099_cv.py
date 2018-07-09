#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:02:36 2018

@author: Kazuki
"""


import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/kazuki_onodera/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from glob import glob
import count
import os
import utils
#utils.start(__file__)
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


use_files = ['train_f0']


# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


# nejumi
files = sorted(glob('../feature_nejumi/*train*'))
X['CNT_PAYMENT'] = np.load(files[0])
X['nejumi_v2'] = np.load(files[1])

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

X = X.rank(method='dense')
gc.collect()


# =============================================================================
# cv bench
# =============================================================================
dtrain = lgb.Dataset(X.iloc[:,:-2], y, 
                     categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(bench): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)



# =============================================================================
# cv with CNT_PAYMENT
# =============================================================================
dtrain = lgb.Dataset(X.iloc[:,:-1], y, 
                     categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(with CNT_PAYMENT): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


# =============================================================================
# cv with CNT_PAYMENT + nejumi_v3
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(with CNT_PAYMENT + nejumi_v3): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


#==============================================================================
utils.end(__file__)
"""
In [2]: files
Out[2]: 
['../feature/train_f001_AMT_ANNUITY-d-CNT_FAM_MEMBERS.f',
 '../feature/train_f001_AMT_ANNUITY.f',
 '../feature/train_f001_AMT_CREDIT-d-CNT_FAM_MEMBERS.f',
 '../feature/train_f001_AMT_CREDIT.f',
 '../feature/train_f001_AMT_GOODS_PRICE-d-CNT_FAM_MEMBERS.f',
 '../feature/train_f001_AMT_GOODS_PRICE.f',
 '../feature/train_f001_AMT_INCOME_TOTAL-d-CNT_FAM_MEMBERS.f',
 '../feature/train_f001_AMT_INCOME_TOTAL.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_DAY.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_HOUR.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_MON.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_QRT.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_WEEK.f',
 '../feature/train_f001_AMT_REQ_CREDIT_BUREAU_YEAR.f',
 '../feature/train_f001_APARTMENTS_AVG.f',
 '../feature/train_f001_APARTMENTS_MEDI.f',
 '../feature/train_f001_APARTMENTS_MODE.f',
 '../feature/train_f001_BASEMENTAREA_AVG.f',
 '../feature/train_f001_BASEMENTAREA_MEDI.f',
 '../feature/train_f001_BASEMENTAREA_MODE.f',
 '../feature/train_f001_CNT_CHILDREN-d-CNT_FAM_MEMBERS.f',
 '../feature/train_f001_CNT_FAM_MEMBERS.f',
 '../feature/train_f001_CODE_GENDER.f',
 '../feature/train_f001_COMMONAREA_AVG.f',
 '../feature/train_f001_COMMONAREA_MEDI.f',
 '../feature/train_f001_COMMONAREA_MODE.f',
 '../feature/train_f001_DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED-d-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_REGISTRATION-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH-d-DAYS_REGISTRATION-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_EMPLOYED-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_EMPLOYED_PERC.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-d-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-d-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-d-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH-d-DAYS_REGISTRATION-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_EMPLOYED-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_REGISTRATION-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_REGISTRATION-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-d-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-d-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-d-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-d-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH-d-DAYS_REGISTRATION-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_LAST_PHONE_CHANGE.f',
 '../feature/train_f001_DAYS_REGISTRATION-d-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_REGISTRATION-d-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH-d-DAYS_REGISTRATION-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_BIRTH.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED-d-DAYS_ID_PUBLISH-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED-d-DAYS_ID_PUBLISH-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED-d-DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION.f',
 '../feature/train_f001_DAYS_REGISTRATION-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_DAYS_REGISTRATION.f',
 '../feature/train_f001_DEF_30_CNT_SOCIAL_CIRCLE.f',
 '../feature/train_f001_DEF_60_CNT_SOCIAL_CIRCLE.f',
 '../feature/train_f001_ELEVATORS_AVG.f',
 '../feature/train_f001_ELEVATORS_MEDI.f',
 '../feature/train_f001_ENTRANCES_AVG.f',
 '../feature/train_f001_ENTRANCES_MEDI.f',
 '../feature/train_f001_ENTRANCES_MODE.f',
 '../feature/train_f001_EXT_SOURCES_1-2-3.f',
 '../feature/train_f001_EXT_SOURCES_1-2.f',
 '../feature/train_f001_EXT_SOURCES_1-3.f',
 '../feature/train_f001_EXT_SOURCES_2-1-3.f',
 '../feature/train_f001_EXT_SOURCES_2-3.f',
 '../feature/train_f001_EXT_SOURCES_mean.f',
 '../feature/train_f001_EXT_SOURCES_prod.f',
 '../feature/train_f001_EXT_SOURCES_std.f',
 '../feature/train_f001_EXT_SOURCES_sum.f',
 '../feature/train_f001_EXT_SOURCE_1.f',
 '../feature/train_f001_EXT_SOURCE_2.f',
 '../feature/train_f001_EXT_SOURCE_3.f',
 '../feature/train_f001_FLAG_DOCUMENT_3.f',
 '../feature/train_f001_FLAG_DOCUMENT_6.f',
 '../feature/train_f001_FLAG_DOCUMENT_8.f',
 '../feature/train_f001_FLAG_OWN_CAR.f',
 '../feature/train_f001_FLAG_OWN_REALTY.f',
 '../feature/train_f001_FLAG_WORK_PHONE.f',
 '../feature/train_f001_FLOORSMAX_AVG.f',
 '../feature/train_f001_FLOORSMAX_MEDI.f',
 '../feature/train_f001_FLOORSMAX_MODE.f',
 '../feature/train_f001_FLOORSMIN_AVG.f',
 '../feature/train_f001_FLOORSMIN_MEDI.f',
 '../feature/train_f001_FLOORSMIN_MODE.f',
 '../feature/train_f001_FONDKAPREMONT_MODE.f',
 '../feature/train_f001_HOUR_APPR_PROCESS_START.f',
 '../feature/train_f001_HOUSETYPE_MODE.f',
 '../feature/train_f001_INCOME_PER_PERSON.f',
 '../feature/train_f001_LANDAREA_AVG.f',
 '../feature/train_f001_LANDAREA_MEDI.f',
 '../feature/train_f001_LANDAREA_MODE.f',
 '../feature/train_f001_LIVE_CITY_NOT_WORK_CITY.f',
 '../feature/train_f001_LIVE_REGION_NOT_WORK_REGION.f',
 '../feature/train_f001_LIVINGAPARTMENTS_AVG.f',
 '../feature/train_f001_LIVINGAPARTMENTS_MEDI.f',
 '../feature/train_f001_LIVINGAPARTMENTS_MODE.f',
 '../feature/train_f001_LIVINGAREA_AVG.f',
 '../feature/train_f001_LIVINGAREA_MEDI.f',
 '../feature/train_f001_LIVINGAREA_MODE.f',
 '../feature/train_f001_NAME_CONTRACT_TYPE.f',
 '../feature/train_f001_NAME_EDUCATION_TYPE.f',
 '../feature/train_f001_NAME_FAMILY_STATUS.f',
 '../feature/train_f001_NAME_HOUSING_TYPE.f',
 '../feature/train_f001_NAME_INCOME_TYPE.f',
 '../feature/train_f001_NAME_TYPE_SUITE.f',
 '../feature/train_f001_NEW_ANNUITY_TO_INCOME_RATIO.f',
 '../feature/train_f001_NEW_CAR_TO_BIRTH_RATIO.f',
 '../feature/train_f001_NEW_CAR_TO_EMPLOY_RATIO.f',
 '../feature/train_f001_NEW_INC_BY_ORG.f',
 '../feature/train_f001_NEW_INC_PER_CHLD.f',
 '../feature/train_f001_NEW_LIVE_IND_SUM.f',
 '../feature/train_f001_NEW_PHONE_TO_BIRTH_RATIO.f',
 '../feature/train_f001_NEW_PHONE_TO_EMPLOYED_RATIO.f',
 '../feature/train_f001_NONLIVINGAPARTMENTS_AVG.f',
 '../feature/train_f001_NONLIVINGAPARTMENTS_MEDI.f',
 '../feature/train_f001_NONLIVINGAPARTMENTS_MODE.f',
 '../feature/train_f001_NONLIVINGAREA_AVG.f',
 '../feature/train_f001_NONLIVINGAREA_MEDI.f',
 '../feature/train_f001_NONLIVINGAREA_MODE.f',
 '../feature/train_f001_OBS_30_CNT_SOCIAL_CIRCLE.f',
 '../feature/train_f001_OBS_60_CNT_SOCIAL_CIRCLE.f',
 '../feature/train_f001_OCCUPATION_TYPE.f',
 '../feature/train_f001_ORGANIZATION_TYPE.f',
 '../feature/train_f001_OWN_CAR_AGE-d-DAYS_BIRTH.f',
 '../feature/train_f001_OWN_CAR_AGE-d-DAYS_EMPLOYED.f',
 '../feature/train_f001_OWN_CAR_AGE-m-DAYS_BIRTH.f',
 '../feature/train_f001_OWN_CAR_AGE-m-DAYS_EMPLOYED.f',
 '../feature/train_f001_OWN_CAR_AGE.f',
 '../feature/train_f001_PAYMENT_RATE.f',
 '../feature/train_f001_REGION_POPULATION_RELATIVE.f',
 '../feature/train_f001_REGION_RATING_CLIENT.f',
 '../feature/train_f001_REGION_RATING_CLIENT_W_CITY.f',
 '../feature/train_f001_REG_CITY_NOT_LIVE_CITY.f',
 '../feature/train_f001_REG_CITY_NOT_WORK_CITY.f',
 '../feature/train_f001_TOTALAREA_MODE.f',
 '../feature/train_f001_WALLSMATERIAL_MODE.f',
 '../feature/train_f001_WEEKDAY_APPR_PROCESS_START.f',
 '../feature/train_f001_YEARS_BEGINEXPLUATATION_AVG.f',
 '../feature/train_f001_YEARS_BEGINEXPLUATATION_MEDI.f',
 '../feature/train_f001_YEARS_BEGINEXPLUATATION_MODE.f',
 '../feature/train_f001_YEARS_BUILD_AVG.f',
 '../feature/train_f001_YEARS_BUILD_MEDI.f',
 '../feature/train_f001_YEARS_BUILD_MODE.f',
 '../feature/train_f001_age_finish_payment.f',
 '../feature/train_f001_alldocs_kurt.f',
 '../feature/train_f001_alldocs_mean.f',
 '../feature/train_f001_alldocs_skew.f',
 '../feature/train_f001_alldocs_std.f',
 '../feature/train_f001_annuity-d-CNT_CHILDREN.f',
 '../feature/train_f001_annuity-d-cnt_adults.f',
 '../feature/train_f001_annuity-d-income.f',
 '../feature/train_f001_building_score_avg_mean.f',
 '../feature/train_f001_building_score_avg_std.f',
 '../feature/train_f001_building_score_avg_sum.f',
 '../feature/train_f001_building_score_medi_mean.f',
 '../feature/train_f001_building_score_medi_std.f',
 '../feature/train_f001_building_score_medi_sum.f',
 '../feature/train_f001_building_score_mode_mean.f',
 '../feature/train_f001_building_score_mode_std.f',
 '../feature/train_f001_building_score_mode_sum.f',
 '../feature/train_f001_cnt_adults.f',
 '../feature/train_f001_credit-d-CNT_CHILDREN.f',
 '../feature/train_f001_credit-d-annuity.f',
 '../feature/train_f001_credit-d-cnt_adults.f',
 '../feature/train_f001_credit-d-income.f',
 '../feature/train_f001_goods_price-d-CNT_CHILDREN.f',
 '../feature/train_f001_goods_price-d-annuity.f',
 '../feature/train_f001_goods_price-d-cnt_adults.f',
 '../feature/train_f001_goods_price-d-credit.f',
 '../feature/train_f001_goods_price-d-income.f',
 '../feature/train_f001_goods_price-m-credit-d-income.f',
 '../feature/train_f001_goods_price-m-credit.f',
 '../feature/train_f001_income-d-CNT_CHILDREN.f',
 '../feature/train_f001_income-d-cnt_adults.f',
 '../feature/train_f001_income_per_adult.f',
 '../feature/train_f001_maxwell_feature_1.f']

"""