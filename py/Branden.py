#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:07:35 2018

@author: kazuki.onodera
"""


import pandas as pd
import utils


df = pd.read_csv('../feature_someone/branden/lgb3g_used_features.csv.gz')

df.drop('SK_ID_CURR', axis=1, inplace=True)

X_train = df.iloc[:307511].reset_index(drop=True)
X_test = df.iloc[307511:].reset_index(drop=True)

X_train.add_prefix('Bra_').to_feather('../feature_someone/branden/X_train.f')
X_test.add_prefix('Bra_').to_feather('../feature_someone/branden/X_test.f')



category_branden = ['papp_max_SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_int',
 'papp_min_SK_ID_CURR_PRODUCT_COMBINATION_int',
 'WEEKDAY_APPR_PROCESS_START_int',
 'papp_max_SK_ID_CURR_NAME_TYPE_SUITE_int',
 'papp_min_SK_ID_CURR_NAME_GOODS_CATEGORY_int',
 'papp_min_SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_int',
 'papp_min_SK_ID_CURR_NAME_SELLER_INDUSTRY_int',
 'CODE_GENDER_int',
 'papp_min_SK_ID_CURR_CODE_REJECT_REASON_int',
 'papp_max_SK_ID_CURR_PRODUCT_COMBINATION_int',
 'papp_min_SK_ID_CURR_NAME_CONTRACT_STATUS_int',
 'NAME_FAMILY_STATUS_int',
 'papp_max_SK_ID_CURR_NAME_GOODS_CATEGORY_int',
 'OCCUPATION_TYPE_int',
 'papp_max_SK_ID_CURR_NAME_SELLER_INDUSTRY_int',
 'papp_max_SK_ID_CURR_CODE_REJECT_REASON_int',
 'papp_min_SK_ID_CURR_NAME_YIELD_GROUP_int',
 'papp_min_SK_ID_CURR_NAME_PRODUCT_TYPE_int',
 'papp_max_SK_ID_CURR_CHANNEL_TYPE_int',
 'WALLSMATERIAL_MODE_int',
 'ORGANIZATION_TYPE_int',
 'papp_min_SK_ID_CURR_NAME_CLIENT_TYPE_int',
 'papp_min_SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_int',
 'papp_max_SK_ID_CURR_NAME_CONTRACT_STATUS_int',
 'papp_min_SK_ID_CURR_NAME_PORTFOLIO_int',
 'papp_max_SK_ID_CURR_NAME_YIELD_GROUP_int',
 'papp_min_SK_ID_CURR_CHANNEL_TYPE_int',
 'NAME_EDUCATION_TYPE_int',
 'FONDKAPREMONT_MODE_int',
 'papp_max_SK_ID_CURR_NAME_PRODUCT_TYPE_int',
 'NAME_INCOME_TYPE_int',
 'papp_min_SK_ID_CURR_NAME_CONTRACT_TYPE_int',
 'papp_min_SK_ID_CURR_NAME_TYPE_SUITE_int',
 'NAME_TYPE_SUITE_int',
 'NAME_HOUSING_TYPE_int',
 'papp_max_SK_ID_CURR_NAME_PAYMENT_TYPE_int',
 'papp_max_SK_ID_CURR_NAME_CLIENT_TYPE_int',
 'papp_min_SK_ID_CURR_NAME_PAYMENT_TYPE_int',
 'papp_max_SK_ID_CURR_NAME_CONTRACT_TYPE_int',
 'papp_max_SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_int']



