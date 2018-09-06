#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:58:24 2018

@author: kazuki.onodera

https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb

"""

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score

#import utils
#utils.start(__file__)
#==============================================================================


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

def UseGPFeatures(data):
    v = pd.DataFrame()
    v["i0"] = np.tanh(((((((data["te_ORGANIZATION_TYPE"]) - (((data["EXT_SOURCE_3"]) * 2.0)))) - (((data["EXT_SOURCE_2"]) - ((-1.0*((data["EXT_SOURCE_2"])))))))) - (data["EXT_SOURCE_1"]))) 
    v["i1"] = np.tanh(((((data["NAME_CONTRACT_STATUS_Refused"]) / 2.0)) + (((((((np.minimum(((data["te_OCCUPATION_TYPE"])), ((data["DAYS_BIRTH"])))) - (data["EXT_SOURCE_3"]))) * 2.0)) - (((((data["EXT_SOURCE_2"]) * 2.0)) + (data["EXT_SOURCE_3"]))))))) 
    v["i2"] = np.tanh(((((((-3.0) * (data["EXT_SOURCE_2"]))) - (data["EXT_SOURCE_3"]))) + (((np.tanh((data["te_OCCUPATION_TYPE"]))) - (((data["EXT_SOURCE_1"]) + (((data["EXT_SOURCE_3"]) - (0.0))))))))) 
    v["i3"] = np.tanh(((data["te_OCCUPATION_TYPE"]) + (((((((data["NAME_CONTRACT_STATUS_Refused"]) - ((((((data["EXT_SOURCE_1"]) + (np.minimum(((data["EXT_SOURCE_3"])), ((data["EXT_SOURCE_3"])))))/2.0)) * 2.0)))) - (data["EXT_SOURCE_2"]))) * 2.0)))) 
    v["i4"] = np.tanh(((((((np.tanh((data["NAME_CONTRACT_STATUS_Refused"]))) - (data["EXT_SOURCE_3"]))) - (data["EXT_SOURCE_2"]))) + (((np.minimum(((((data["EXT_SOURCE_3"]) * (-3.0)))), ((data["ca__Active"])))) - (data["EXT_SOURCE_2"]))))) 
    v["i5"] = np.tanh(((data["te_ORGANIZATION_TYPE"]) + (((((((data["NAME_CONTRACT_STATUS_Refused"]) - (data["EXT_SOURCE_2"]))) - (np.where(data["EXT_SOURCE_1"]>0, (9.09630203247070312), data["ca__Closed"] )))) + (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))))) 
    v["i6"] = np.tanh(((((data["te_OCCUPATION_TYPE"]) + ((((((((-1.0*((data["EXT_SOURCE_2"])))) + (data["te_NAME_EDUCATION_TYPE"]))) - (((data["EXT_SOURCE_3"]) + (data["EXT_SOURCE_1"]))))) + (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))))) - (data["EXT_SOURCE_3"]))) 
    v["i7"] = np.tanh(((((data["NAME_PRODUCT_TYPE_walk_in"]) + (((((((np.tanh(((-1.0*((((2.0) - (data["cc_bal_AMT_TOTAL_RECEIVABLE"])))))))) + (data["cc_bal_AMT_TOTAL_RECEIVABLE"]))) - (data["EXT_SOURCE_2"]))) - (data["EXT_SOURCE_3"]))))) * 2.0)) 
    v["i8"] = np.tanh(((((((((((((data["te_NAME_EDUCATION_TYPE"]) - (data["CODE_GENDER"]))) - (data["DAYS_FIRST_DRAWING"]))) * 2.0)) * 2.0)) - (((data["EXT_SOURCE_3"]) * 2.0)))) * 2.0)) 
    v["i9"] = np.tanh((((((((data["REGION_RATING_CLIENT"]) + (data["te_ORGANIZATION_TYPE"]))/2.0)) + (((np.maximum(((-3.0)), ((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"])))) + (np.minimum(((data["DAYS_CREDIT"])), ((((data["cnt_FLAG_DOCUMENT_3"]) - (data["EXT_SOURCE_2"])))))))))) * 2.0)) 
    v["i10"] = np.tanh((((((((7.0)) * (data["ty__Microloan"]))) - (data["EXT_SOURCE_3"]))) + ((((((data["CNT_INSTALMENT_FUTURE"]) * (data["CNT_INSTALMENT_FUTURE"]))) + (((data["EXT_SOURCE_1"]) * (((-1.0) * 2.0)))))/2.0)))) 
    v["i11"] = np.tanh((((((-1.0*((((data["EXT_SOURCE_3"]) - (0.0)))))) + (data["NAME_YIELD_GROUP_high"]))) - ((-1.0*((np.where(((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) / 2.0)) / 2.0)>0, (8.0), data["te_FLAG_OWN_CAR"] ))))))) 
    v["i12"] = np.tanh(((((data["CNT_PAYMENT"]) - (((((((data["EXT_SOURCE_1"]) + (data["SK_ID_PREV_y"]))) - ((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (((data["CNT_PAYMENT"]) - (data["EXT_SOURCE_2"]))))/2.0)))) * 2.0)))) + (data["NAME_CLIENT_TYPE_New"]))) 
    v["i13"] = np.tanh(((((((np.where(((((2.0) / 2.0)) - (data["EXT_SOURCE_2"]))>0, data["te_CODE_GENDER"], -2.0 )) + (data["te_NAME_CONTRACT_TYPE"]))) * 2.0)) + (data["cnt_CODE_GENDER"]))) 
    v["i14"] = np.tanh(((((((((data["REG_CITY_NOT_LIVE_CITY"]) + (((((data["ty__Microloan"]) * 2.0)) * 2.0)))) + (np.maximum(((data["cc_bal_CNT_DRAWINGS_CURRENT"])), ((((data["NAME_CONTRACT_STATUS_Refused"]) - (data["cnt_REG_CITY_NOT_LIVE_CITY"])))))))) + (data["cc_bal_CNT_DRAWINGS_CURRENT"]))) * 2.0)) 
    v["i15"] = np.tanh((((((((((data["te_NAME_EDUCATION_TYPE"]) - (((data["AMT_ANNUITY"]) - (((data["CNT_INSTALMENT_FUTURE"]) + (data["ty__Microloan"]))))))) + (data["DEF_30_CNT_SOCIAL_CIRCLE"]))/2.0)) * 2.0)) * 2.0)) 
    v["i16"] = np.tanh(((data["te_FLAG_DOCUMENT_3"]) + (((((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) * 2.0)) + (((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) * 2.0)) + (((0.0) + (data["DAYS_ID_PUBLISH"]))))))) - (data["cnt_NAME_FAMILY_STATUS"]))))) 
    v["i17"] = np.tanh(((((data["CNT_INSTALMENT_FUTURE"]) + (((data["NAME_CLIENT_TYPE_New"]) + (((data["avg_buro_buro_bal_status_1"]) + (((data["te_WALLSMATERIAL_MODE"]) - (data["PRODUCT_COMBINATION_POS_industry_with_interest"]))))))))) + (((data["NAME_CLIENT_TYPE_New"]) - (data["DAYS_FIRST_DRAWING"]))))) 
    v["i18"] = np.tanh(((np.where(data["NAME_YIELD_GROUP_low_action"]>0, -3.0, ((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["CNT_PAYMENT"]) + (np.maximum(((-3.0)), ((data["NAME_YIELD_GROUP_high"]))))))) )) + (data["ty__Microloan"]))) 
    v["i19"] = np.tanh(((((data["AMT_ANNUITY_x"]) * 2.0)) + (((((((data["AMT_ANNUITY_x"]) * 2.0)) + (np.maximum(((data["NAME_YIELD_GROUP_XNA"])), ((((data["DAYS_LAST_PHONE_CHANGE"]) / 2.0))))))) + (((data["PRODUCT_COMBINATION_Cash_X_Sell__low"]) * (-3.0))))))) 
    v["i20"] = np.tanh(((((data["cnt_FLAG_OWN_CAR"]) - (((data["inst_NUM_INSTALMENT_VERSION"]) - (np.where(data["EXT_SOURCE_1"]>0, -3.0, data["CODE_REJECT_REASON_SCOFR"] )))))) - (data["RATE_DOWN_PAYMENT"]))) 
    v["i21"] = np.tanh(((((((data["OWN_CAR_AGE"]) - (((((data["CODE_GENDER"]) - (((data["OWN_CAR_AGE"]) + (data["OWN_CAR_AGE"]))))) / 2.0)))) * 2.0)) + (np.maximum(((data["CNT_PAYMENT"])), ((data["ty__Microloan"])))))) 
    v["i22"] = np.tanh(((((((data["DEF_60_CNT_SOCIAL_CIRCLE"]) - (((data["NAME_YIELD_GROUP_low_action"]) + ((((data["avg_buro_buro_bal_status_1"]) < (data["PRODUCT_COMBINATION_Cash_Street__low"]))*1.)))))) + (data["avg_buro_buro_bal_status_1"]))) - (data["AMT_ANNUITY"]))) 
    v["i23"] = np.tanh(np.where(((data["ty__Mortgage"]) * 2.0)>0, -3.0, (((data["cnt_DAYS_BIRTH"]) + ((((((data["NAME_PORTFOLIO_XNA"]) + (np.maximum(((data["CODE_REJECT_REASON_HC"])), ((0.0)))))/2.0)) - (data["CHANNEL_TYPE_Channel_of_corporate_sales"]))))/2.0) )) 
    v["i24"] = np.tanh(((((((data["DAYS_LAST_PHONE_CHANGE"]) + (data["nans"]))) + (((((((data["SK_DPD"]) + (data["CNT_PAYMENT"]))) / 2.0)) * 2.0)))) + (((((-1.0*((data["PRODUCT_COMBINATION_Cash_Street__low"])))) + (data["cnt_ORGANIZATION_TYPE"]))/2.0)))) 
    v["i25"] = np.tanh(np.where(data["inst_AMT_PAYMENT"]>0, (-1.0*((data["Active"]))), (((-1.0) > (((np.minimum((((1.0))), ((data["inst_AMT_PAYMENT"])))) * 2.0)))*1.) )) 
    v["i26"] = np.tanh(((np.where(np.tanh((np.where((10.0)>0, data["cc_bal_AMT_RECIVABLE"], data["cc_bal_AMT_BALANCE"] )))>0, data["cc_bal_CNT_DRAWINGS_CURRENT"], ((data["SK_DPD"]) * ((6.0))) )) + ((-1.0*((data["FLAG_DOCUMENT_18"])))))) 
    v["i27"] = np.tanh((((((((((((data["cnt_FLAG_DOCUMENT_16"]) + (data["AMT_ANNUITY_x"]))/2.0)) - (data["ty__Mortgage"]))) + (data["ty__Microloan"]))) * 2.0)) + (np.tanh(((((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) > (data["ty__Microloan"]))*1.)))))) 
    v["i28"] = np.tanh(np.where(data["ca__Sold"]>0, 2.0, (((-1.0*((((data["RATE_DOWN_PAYMENT"]) + (((data["AMT_GOODS_PRICE_x"]) * (((((data["AMT_GOODS_PRICE_x"]) * 2.0)) - (-1.0)))))))))) + (data["cnt_NAME_INCOME_TYPE"])) )) 
    v["i29"] = np.tanh(((((((3.0) * ((((data["CHANNEL_TYPE_AP___Cash_loan_"]) > (((data["inst_AMT_PAYMENT"]) / 2.0)))*1.)))) * 2.0)) - ((-1.0*((np.minimum(((data["cnt_WEEKDAY_APPR_PROCESS_START"])), (((((data["CHANNEL_TYPE_AP___Cash_loan_"]) > (data["MONTHS_BALANCE"]))*1.)))))))))) 
    v["i30"] = np.tanh(((((((np.maximum(((data["CNT_INSTALMENT_FUTURE"])), (((((((-1.0*(((((((data["te_FLAG_DOCUMENT_6"]) * 2.0)) + (data["CNT_INSTALMENT_FUTURE"]))/2.0))))) * 2.0)) * 2.0))))) - (((data["te_FLAG_DOCUMENT_6"]) * 2.0)))) / 2.0)) * 2.0)) 
    v["i31"] = np.tanh((((((data["te_NAME_FAMILY_STATUS"]) - (data["PRODUCT_COMBINATION_Cash_Street__low"]))) + (np.minimum((((((-1.0*((data["CHANNEL_TYPE_Channel_of_corporate_sales"])))) - (((((data["AMT_CREDIT_SUM_LIMIT"]) * 2.0)) * 2.0))))), ((((data["DAYS_REGISTRATION"]) / 2.0))))))/2.0)) 
    v["i32"] = np.tanh(((((-1.0*((data["Active"])))) + ((((((data["HOUSETYPE_MODE"]) - (data["FLAG_DOCUMENT_18"]))) + ((((((data["CODE_REJECT_REASON_SCO"]) + (data["CHANNEL_TYPE_Channel_of_corporate_sales"]))/2.0)) * (-2.0))))/2.0)))/2.0)) 
    v["i33"] = np.tanh((((((((-1.0*((((data["inst_NUM_INSTALMENT_VERSION"]) * (data["inst_NUM_INSTALMENT_VERSION"])))))) + (data["NFLAG_INSURED_ON_APPROVAL"]))/2.0)) + ((((((((data["cnt_FLAG_PHONE"]) + (data["NFLAG_INSURED_ON_APPROVAL"]))/2.0)) - (data["inst_AMT_PAYMENT"]))) / 2.0)))/2.0)) 
    v["i34"] = np.tanh((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) + (np.maximum(((((-1.0) - ((-1.0*((data["FLAG_DOCUMENT_2"]))))))), ((np.where(data["cc_bal_AMT_TOTAL_RECEIVABLE"]>0, data["FLAG_WORK_PHONE"], data["cc_bal_AMT_TOTAL_RECEIVABLE"] ))))))/2.0)) 
    v["i35"] = np.tanh(np.minimum((((((((((data["NAME_PAYMENT_TYPE_XNA"]) / 2.0)) + (data["cnt_FLAG_DOCUMENT_13"]))/2.0)) - (np.tanh(((((((data["cnt_FLAG_DOCUMENT_2"]) < (data["NAME_PAYMENT_TYPE_XNA"]))*1.)) * (data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"])))))))), (((5.0))))) 
    v["i36"] = np.tanh(((((-1.0*((((data["DAYS_ENDDATE_FACT"]) + (np.minimum(((data["YEARS_BUILD_MODE"])), (((11.45870018005371094)))))))))) + (((((-1.0*((((data["avg_buro_buro_bal_status_0"]) + (data["FLAG_PHONE"])))))) + ((-1.0*(((0.0))))))/2.0)))/2.0)) 
    v["i37"] = np.tanh(((np.minimum((((((-2.0) > (data["EXT_SOURCE_3"]))*1.))), ((((data["avg_buro_buro_bal_status_1"]) - (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_QRT"])), ((((data["EXT_SOURCE_3"]) - (np.tanh((-3.0))))))))))))) * (3.0))) 
    v["i38"] = np.tanh((((data["SK_DPD"]) > ((((data["ty__Microloan"]) + (np.where((-1.0*((data["NAME_GOODS_CATEGORY_Furniture"])))>0, (((data["te_REG_REGION_NOT_LIVE_REGION"]) + (((0.0) / 2.0)))/2.0), (((-1.0*((data["ty__Microloan"])))) / 2.0) )))/2.0)))*1.)) 
    v["i39"] = np.tanh(((np.where((((((data["NAME_GOODS_CATEGORY_Insurance"]) - (data["cnt_FLAG_DOCUMENT_16"]))) > (data["ty__Microloan"]))*1.)>0, -2.0, data["ty__Microloan"] )) - (np.tanh((((data["cnt_FLAG_DOCUMENT_2"]) * 2.0)))))) 
    v["i40"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["NAME_FAMILY_STATUS"], (((((((2.0)) + (np.where(np.minimum(((-3.0)), ((data["FLAG_DOCUMENT_2"])))>0, data["cnt_FLAG_DOCUMENT_2"], data["NAME_YIELD_GROUP_middle"] )))/2.0)) < (data["NAME_YIELD_GROUP_middle"]))*1.) )) 
    v["i41"] = np.tanh(np.where((((data["DAYS_LAST_DUE"]) < (data["RATE_DOWN_PAYMENT"]))*1.)>0, np.minimum(((0.0)), (((-1.0*((data["RATE_DOWN_PAYMENT"])))))), (((((data["DAYS_LAST_DUE"]) + (data["te_CNT_CHILDREN"]))/2.0)) / 2.0) )) 
    v["i42"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], ((data["AMT_ANNUITY_x"]) - ((((((data["AMT_ANNUITY_x"]) + (data["cnt_FLAG_DOCUMENT_2"]))/2.0)) * (((data["inst_DAYS_INSTALMENT"]) + (data["AMT_ANNUITY_x"])))))) )) 
    v["i43"] = np.tanh(((((((-1.0*((data["AMT_GOODS_PRICE_x"])))) < ((((data["CHANNEL_TYPE_Channel_of_corporate_sales"]) > (((data["AMT_GOODS_PRICE_x"]) / 2.0)))*1.)))*1.)) * (np.where(data["AMT_GOODS_PRICE_x"]>0, data["FLAG_DOCUMENT_2"], 3.0 )))) 
    v["i44"] = np.tanh((-1.0*(((((((((((((-1.0*((((np.tanh((((-1.0) * 2.0)))) / 2.0))))) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) + (data["DAYS_EMPLOYED"]))) < (data["CODE_REJECT_REASON_SCO"]))*1.)) / 2.0))))) 
    v["i45"] = np.tanh((((((data["inst_DAYS_INSTALMENT"]) < (data["inst_DAYS_ENTRY_PAYMENT"]))*1.)) + (np.minimum(((((data["inst_DAYS_INSTALMENT"]) * (-1.0)))), ((data["AMT_CREDIT_x"])))))) 
    v["i46"] = np.tanh(np.where(data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]>0, (-1.0*((np.maximum((((2.0))), ((((0.0) * 2.0))))))), (((((data["DAYS_ID_PUBLISH"]) - ((-1.0*((data["DAYS_DECISION"])))))) > (2.0))*1.) )) 
    v["i47"] = np.tanh(np.where((((np.maximum(((data["NAME_FAMILY_STATUS"])), ((((data["NAME_TYPE_SUITE_Unaccompanied"]) * 2.0))))) < (-1.0))*1.)>0, 2.0, (((np.where(data["NAME_FAMILY_STATUS"]>0, -1.0, data["NAME_GOODS_CATEGORY_Direct_Sales"] )) + (data["NAME_FAMILY_STATUS"]))/2.0) )) 
    v["i48"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 1.0, np.minimum(((data["cc_bal_AMT_BALANCE"])), ((np.where(data["cc_bal_SK_DPD_DEF"]>0, np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((3.0))), (-1.0*((data["CHANNEL_TYPE_Channel_of_corporate_sales"]))) )))) )) 
    v["i49"] = np.tanh(((((np.where(((np.minimum(((data["cnt_FLAG_DOCUMENT_13"])), ((data["FLAG_DOCUMENT_2"])))) - (data["cc_bal_SK_DPD_DEF"]))>0, ((data["CNT_INSTALMENT_FUTURE"]) * 2.0), data["cnt_FLAG_DOCUMENT_13"] )) - (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) + (data["NAME_GOODS_CATEGORY_Insurance"]))) 
    v["i50"] = np.tanh(np.minimum((((((-1.0*((data["NAME_GOODS_CATEGORY_Furniture"])))) - (data["inst_AMT_PAYMENT"])))), (((((((data["inst_AMT_PAYMENT"]) / 2.0)) < (data["NAME_GOODS_CATEGORY_Furniture"]))*1.))))) 
    v["i51"] = np.tanh((((((np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((data["AMT_GOODS_PRICE_y"])))) / 2.0)) > (((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]) * (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"]))))*1.)) 
    v["i52"] = np.tanh((-1.0*((np.where(data["CNT_INSTALMENT"]>0, data["Active"], np.minimum(((data["SK_ID_PREV_y"])), ((2.0))) ))))) 
    v["i53"] = np.tanh(np.where(data["CNT_INSTALMENT_FUTURE"]>0, ((data["SK_DPD"]) + (np.minimum(((data["CNT_INSTALMENT_FUTURE"])), ((data["SK_ID_PREV_x"]))))), np.tanh(((-1.0*(((((data["SK_ID_PREV_x"]) > (data["NAME_PRODUCT_TYPE_x_sell"]))*1.)))))) )) 
    v["i54"] = np.tanh(np.tanh(((((((((data["SK_DPD"]) - (data["FLAG_DOCUMENT_18"]))) * 2.0)) > ((((data["NAME_GOODS_CATEGORY_Insurance"]) + ((-1.0*((data["SK_DPD"])))))/2.0)))*1.)))) 
    v["i55"] = np.tanh(np.where((-1.0*((data["SK_DPD"])))>0, ((np.minimum((((-1.0*(((((data["PRODUCT_COMBINATION_Cash_X_Sell__low"]) + (data["FLAG_DOCUMENT_16"]))/2.0)))))), (((-1.0*((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]))))))) / 2.0), -2.0 )) 
    v["i56"] = np.tanh((((((data["cnt_FLAG_DOCUMENT_2"]) + ((((data["SK_DPD"]) + (((data["cnt_FLAG_DOCUMENT_2"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"]))))/2.0)))) > (((data["FLAG_DOCUMENT_2"]) * (2.0))))*1.)) 
    v["i57"] = np.tanh((((np.where(np.where(data["CNT_INSTALMENT_FUTURE"]>0, -2.0, data["SK_DPD"] )>0, -2.0, data["SK_DPD"] )) + ((((data["SK_DPD"]) + ((-1.0*((data["te_REG_REGION_NOT_LIVE_REGION"])))))/2.0)))/2.0)) 
    v["i58"] = np.tanh((-1.0*((np.maximum(((((((((((((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]) * 2.0)) + (data["ty__Car_loan"]))/2.0)) + (data["inst_NUM_INSTALMENT_VERSION"]))/2.0)) / 2.0))), ((data["ty__Mortgage"]))))))) 
    v["i59"] = np.tanh(((((((((((((data["NAME_GOODS_CATEGORY_Insurance"]) + (((data["DAYS_REGISTRATION"]) / 2.0)))/2.0)) / 2.0)) + ((((data["DAYS_REGISTRATION"]) + (data["CNT_INSTALMENT_FUTURE"]))/2.0)))) - (2.0))) > ((-1.0*((data["DAYS_REGISTRATION"])))))*1.)) 
    v["i60"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, 1.0, ((((data["FLAG_DOCUMENT_2"]) + ((((((data["DAYS_FIRST_DUE"]) + ((-1.0*((data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])))))) + (data["NAME_GOODS_CATEGORY_Direct_Sales"]))/2.0)))) / 2.0) )) 
    v["i61"] = np.tanh((((13.15089130401611328)) * (np.where(((data["ca__Sold"]) - (data["cc_bal_SK_DPD_DEF"]))>0, np.minimum(((data["FLAG_DOCUMENT_2"])), ((((np.minimum(((((data["cc_bal_cc_bal_status__Sent_proposal"]) * 2.0))), ((data["ca__Sold"])))) * 2.0)))), data["cc_bal_SK_DPD_DEF"] )))) 
    v["i62"] = np.tanh(np.maximum(((np.maximum((((((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) + (data["cc_bal_CNT_DRAWINGS_OTHER_CURRENT"])))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"]))))), (((((((((data["cc_bal_CNT_DRAWINGS_OTHER_CURRENT"]) + (data["ca__Sold"]))) > ((5.0)))*1.)) * 2.0))))) 
    v["i63"] = np.tanh((-1.0*(((((data["CHANNEL_TYPE_Channel_of_corporate_sales"]) + (np.where((-1.0*((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])))>0, data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"], data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"] )))/2.0))))) 
    v["i64"] = np.tanh((((((((6.87359857559204102)) * ((((6.87359476089477539)) * (((((data["cc_bal_AMT_BALANCE"]) - (data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"]))) * (np.where(data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"]>0, 1.0, 3.0 )))))))) * 2.0)) / 2.0)) 
    v["i65"] = np.tanh(((np.minimum(((((1.0) - (data["NAME_YIELD_GROUP_high"])))), (((((((data["NAME_YIELD_GROUP_high"]) > ((((data["NAME_YIELD_GROUP_high"]) > (((data["NAME_YIELD_GROUP_high"]) + (data["cnt_HOUR_APPR_PROCESS_START_x"]))))*1.)))*1.)) * 2.0))))) / 2.0)) 
    v["i66"] = np.tanh(((data["NONLIVINGAPARTMENTS_MODE"]) - ((((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) + ((((data["FLAG_DOCUMENT_14"]) > ((-1.0*((((data["cc_bal_cc_bal_status__Sent_proposal"]) * ((-1.0*(((-1.0*((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])))))))))))))*1.)))) + (data["FLAG_DOCUMENT_14"]))/2.0)))) 
    v["i67"] = np.tanh((((((data["cnt_FLAG_DOCUMENT_13"]) + (np.minimum(((data["cc_bal_SK_ID_PREV"])), ((((((((data["cnt_FLAG_DOCUMENT_18"]) + (data["cc_bal_SK_ID_PREV"]))/2.0)) + (((data["cnt_FLAG_DOCUMENT_18"]) * 2.0)))/2.0))))))) + ((-1.0*((data["cc_bal_cc_bal_status__Sent_proposal"])))))/2.0)) 
    v["i68"] = np.tanh((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) + ((-1.0*((np.where(((((data["AMT_REQ_CREDIT_BUREAU_MON"]) / 2.0)) + (data["AMT_CREDIT_SUM_LIMIT"]))>0, np.maximum(((data["AMT_CREDIT_SUM_LIMIT"])), ((((data["AMT_REQ_CREDIT_BUREAU_MON"]) / 2.0)))), data["FLOORSMAX_MODE"] ))))))/2.0)) 
    v["i69"] = np.tanh((((data["cnt_FLAG_DOCUMENT_2"]) < ((((data["NAME_GOODS_CATEGORY_Insurance"]) + ((((data["DAYS_FIRST_DUE"]) > (((np.tanh(((((1.0) < (data["DAYS_FIRST_DUE"]))*1.)))) * 2.0)))*1.)))/2.0)))*1.)) 
    v["i70"] = np.tanh(((((((data["NAME_GOODS_CATEGORY_Gardening"]) < (data["SK_DPD"]))*1.)) + (np.minimum(((np.where(data["SK_DPD"]>0, data["cnt_ORGANIZATION_TYPE"], data["REGION_POPULATION_RELATIVE"] ))), (((((1.0) > (data["REGION_POPULATION_RELATIVE"]))*1.))))))/2.0)) 
    v["i71"] = np.tanh(((((data["FLAG_DOCUMENT_16"]) * (((data["CODE_REJECT_REASON_SCO"]) + ((-1.0*((data["REGION_RATING_CLIENT_W_CITY"])))))))) * 2.0)) 
    v["i72"] = np.tanh(((np.maximum(((((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (data["te_ORGANIZATION_TYPE"]))/2.0)) + (((data["FLAG_DOCUMENT_2"]) * 2.0)))/2.0))), ((data["cu__currency_3"])))) + ((-1.0*((((((data["NAME_GOODS_CATEGORY_Gardening"]) + (data["cnt_FLAG_DOCUMENT_2"]))) / 2.0))))))) 
    v["i73"] = np.tanh(np.where(np.where(data["cnt_NAME_HOUSING_TYPE"]>0, data["cnt_FLAG_DOCUMENT_16"], data["cnt_NAME_HOUSING_TYPE"] )>0, data["FLAG_DOCUMENT_16"], np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["FLAG_DOCUMENT_16"], data["cnt_FLAG_DOCUMENT_16"] ) )) 
    v["i74"] = np.tanh(np.minimum(((((data["cnt_FLAG_DOCUMENT_16"]) - ((((np.tanh((data["NAME_GOODS_CATEGORY_Clothing_and_Accessories"]))) > (np.tanh((np.minimum((((((data["cnt_FLAG_DOCUMENT_16"]) + (data["cnt_FLAG_DOCUMENT_16"]))/2.0))), ((data["FLAG_DOCUMENT_16"])))))))*1.))))), ((data["te_FLAG_DOCUMENT_16"])))) 
    v["i75"] = np.tanh((((((((-1.0*((np.minimum(((data["AMT_CREDIT_SUM_DEBT"])), (((0.37224778532981873)))))))) * 2.0)) * (data["AMT_CREDIT_SUM_DEBT"]))) - (np.where(data["AMT_CREDIT_SUM_DEBT"]>0, -3.0, (((-1.0*((data["AMT_CREDIT_SUM_DEBT"])))) * 2.0) )))) 
    v["i76"] = np.tanh((((((data["DAYS_EMPLOYED"]) > (np.minimum(((np.tanh((np.minimum(((data["DAYS_EMPLOYED"])), ((((0.0) + ((((-1.0*((1.0)))) / 2.0)))))))))), ((data["avg_buro_buro_bal_status_0"])))))*1.)) / 2.0)) 
    v["i77"] = np.tanh(np.where(data["EXT_SOURCE_3"]>0, data["EXT_SOURCE_3"], ((data["EXT_SOURCE_3"]) * (((data["EXT_SOURCE_3"]) + (2.0)))) )) 
    v["i78"] = np.tanh(((((np.maximum(((np.maximum(((-3.0)), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * 2.0)))))), (((((-1.0*((((data["EXT_SOURCE_3"]) * 2.0))))) + (-3.0)))))) * 2.0)) * 2.0)) 
    v["i79"] = np.tanh(np.minimum(((((((-1.0*((data["ty__Car_loan"])))) + (np.maximum((((((-1.0*((data["inst_AMT_PAYMENT"])))) / 2.0))), ((data["AMT_REQ_CREDIT_BUREAU_DAY"])))))/2.0))), ((((((2.0) * 2.0)) - (data["te_LIVE_REGION_NOT_WORK_REGION"])))))) 
    v["i80"] = np.tanh(((np.minimum(((np.where((((data["te_OCCUPATION_TYPE"]) < (1.0))*1.)>0, data["FLAG_DOCUMENT_2"], np.minimum(((data["AMT_CREDIT_y"])), ((data["AMT_CREDIT_y"]))) ))), ((np.where(data["cc_bal_cc_bal_status__Sent_proposal"]>0, data["FLAG_DOCUMENT_2"], data["AMT_CREDIT_y"] ))))) / 2.0)) 
    v["i81"] = np.tanh((((-1.0*(((((-1.0) > (((data["PRODUCT_COMBINATION_Cash_Street__low"]) * (data["inst_AMT_PAYMENT"]))))*1.))))) - ((((np.maximum(((-2.0)), ((data["PRODUCT_COMBINATION_Cash_Street__low"])))) > (((2.0) * 2.0)))*1.)))) 
    v["i82"] = np.tanh(((np.maximum(((((data["Returned_to_the_store"]) + (np.tanh((((data["cc_bal_cc_bal_status__Sent_proposal"]) * ((((data["Returned_to_the_store"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0))))))))), ((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["te_CNT_CHILDREN"]))))))) * 2.0)) 
    v["i83"] = np.tanh((((((data["cnt_FLAG_DOCUMENT_15"]) + (data["FLAG_DOCUMENT_2"]))/2.0)) + (np.where((((1.49800336360931396)) - (((-3.0) + (data["ca__Sold"]))))>0, data["NAME_GOODS_CATEGORY_Insurance"], data["ca__Sold"] )))) 
    v["i84"] = np.tanh((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) > ((((((3.23952984809875488)) * (((((((data["cnt_FLAG_DOCUMENT_5"]) / 2.0)) - (np.where(data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]>0, data["CODE_REJECT_REASON_SCOFR"], -1.0 )))) * 2.0)))) * 2.0)))*1.)) 
    v["i85"] = np.tanh((((np.tanh((np.tanh(((4.0)))))) + (np.minimum(((data["AMT_CREDIT_x"])), ((((((data["NAME_GOODS_CATEGORY_Insurance"]) - (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) * (2.0)))))))/2.0)) 
    v["i86"] = np.tanh((-1.0*((((((((6.0)) * (data["cc_bal_SK_DPD_DEF"]))) > ((((1.0) + (data["AMT_GOODS_PRICE_x"]))/2.0)))*1.))))) 
    v["i87"] = np.tanh((-1.0*(((((data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]) > ((((((data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]) < (data["inst_AMT_PAYMENT"]))*1.)) * (data["cnt_HOUR_APPR_PROCESS_START_x"]))))*1.))))) 
    v["i88"] = np.tanh(np.tanh((((np.where(((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["AMT_GOODS_PRICE_x"])))) * (data["AMT_CREDIT_y"]))>0, data["cc_bal_CNT_DRAWINGS_CURRENT"], (-1.0*((data["AMT_GOODS_PRICE_x"]))) )) / 2.0)))) 
    v["i89"] = np.tanh(((((((((((2.0) > (data["PRODUCT_COMBINATION_Cash_X_Sell__low"]))*1.)) - (1.0))) + ((((((data["NAME_GOODS_CATEGORY_Insurance"]) > ((((data["SK_DPD"]) > (data["cc_bal_cc_bal_status__Sent_proposal"]))*1.)))*1.)) * 2.0)))/2.0)) * 2.0)) 
    v["i90"] = np.tanh(((((((((((data["SK_DPD"]) * 2.0)) > (data["PRODUCT_COMBINATION_POS_industry_without_interest"]))*1.)) + (np.tanh((np.tanh((np.tanh(((((data["FLAG_DOCUMENT_2"]) + (((((data["HOUSETYPE_MODE"]) / 2.0)) / 2.0)))/2.0)))))))))/2.0)) * 2.0)) 
    v["i91"] = np.tanh(((np.where((((np.maximum(((((2.0) * 2.0))), (((((data["te_FLAG_OWN_REALTY"]) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.))))) < (data["NAME_TYPE_SUITE_Children"]))*1.)>0, data["NAME_TYPE_SUITE_Children"], ((data["CODE_REJECT_REASON_VERIF"]) / 2.0) )) * (data["te_FLAG_OWN_REALTY"]))) 
    v["i92"] = np.tanh(np.minimum((((((((((2.68808555603027344)) < (data["AMT_CREDIT_y"]))*1.)) + (0.0))/2.0))), (((((((2.0) > (data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]))*1.)) - (data["te_NAME_EDUCATION_TYPE"])))))) 
    v["i93"] = np.tanh(((((((data["cu__currency_3"]) * 2.0)) - (data["YEARS_BEGINEXPLUATATION_MODE"]))) * (np.maximum(((data["ORGANIZATION_TYPE"])), (((((-1.0*((((data["YEARS_BEGINEXPLUATATION_MODE"]) + (((data["cc_bal_cc_bal_status__Sent_proposal"]) * 2.0))))))) + (data["ORGANIZATION_TYPE"])))))))) 
    v["i94"] = np.tanh((-1.0*(((((np.maximum((((((data["NAME_GOODS_CATEGORY_Insurance"]) + (np.maximum((((((data["NAME_GOODS_CATEGORY_Insurance"]) + (1.0))/2.0))), ((data["WEEKDAY_APPR_PROCESS_START_TUESDAY"])))))/2.0))), ((data["cc_bal_cc_bal_status__Sent_proposal"])))) + (np.minimum(((data["SK_ID_PREV_y"])), ((0.0)))))/2.0))))) 
    v["i95"] = np.tanh((((((((((np.maximum(((0.0)), ((data["NAME_CLIENT_TYPE_Repeater"])))) < (data["AMT_CREDIT_y"]))*1.)) * (((data["NAME_CLIENT_TYPE_Repeater"]) / 2.0)))) * 2.0)) * (data["AMT_CREDIT_y"]))) 
    v["i96"] = np.tanh(((((-1.0*((data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"])))) + ((-1.0*(((((data["NAME_CONTRACT_STATUS_Refused"]) > (np.where(data["cc_bal_SK_DPD_DEF"]>0, data["CODE_REJECT_REASON_XAP"], 1.0 )))*1.))))))/2.0)) 
    v["i97"] = np.tanh(np.maximum((((((((data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]) * 2.0)) + (np.minimum(((data["SK_ID_PREV_x"])), ((1.0)))))/2.0))), (((((((1.0) - (data["NAME_GOODS_CATEGORY_Insurance"]))) < (((data["cc_bal_AMT_RECIVABLE"]) / 2.0)))*1.))))) 
    v["i98"] = np.tanh(np.minimum(((np.minimum(((data["cc_bal_SK_DPD_DEF"])), ((data["cc_bal_SK_DPD_DEF"]))))), (((-1.0*((np.minimum(((((data["inst_SK_ID_PREV"]) * 2.0))), (((-1.0*((np.tanh(((-1.0*(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (data["DAYS_LAST_DUE_1ST_VERSION"]))/2.0))))))))))))))))))) 
    v["i99"] = np.tanh((-1.0*(((((((data["cc_bal_SK_DPD_DEF"]) < (data["cu__currency_3"]))*1.)) - ((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > (np.tanh(((((((data["cc_bal_SK_DPD_DEF"]) > (-1.0))*1.)) + (1.0))))))*1.))))))) 
    v["i100"] = np.tanh(((np.minimum(((((data["NONLIVINGAPARTMENTS_MODE"]) * (np.maximum(((data["NAME_TYPE_SUITE_Unaccompanied"])), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"])))) * (((data["cc_bal_SK_DPD_DEF"]) + (((data["NAME_TYPE_SUITE_Unaccompanied"]) * 2.0)))))) 
    v["i101"] = np.tanh(((data["cu__currency_3"]) + (((((data["inst_AMT_PAYMENT"]) * (((data["FLAG_DOCUMENT_14"]) - (np.maximum(((((data["cu__currency_3"]) - (data["inst_AMT_PAYMENT"])))), ((data["cc_bal_SK_DPD_DEF"])))))))) - (data["cnt_FLAG_DOCUMENT_2"]))))) 
    v["i102"] = np.tanh(np.tanh((((((data["EXT_SOURCE_2"]) * (data["EXT_SOURCE_2"]))) + (((data["EXT_SOURCE_2"]) * 2.0)))))) 
    v["i103"] = np.tanh((-1.0*((((np.minimum((((0.67251223325729370))), ((((((((data["EXT_SOURCE_2"]) + ((-1.0*((-2.0)))))) * 2.0)) * 2.0))))) / 2.0))))) 
    v["i104"] = np.tanh(((np.where(data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]>0, data["AMT_CREDIT_y"], (((((data["HOUR_APPR_PROCESS_START_y"]) < (np.tanh((-2.0))))*1.)) / 2.0) )) - ((((1.0) < (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]))*1.)))) 
    v["i105"] = np.tanh((((data["cc_bal_SK_DPD_DEF"]) + (((data["cnt_FLAG_DOCUMENT_14"]) + (np.maximum(((((((np.tanh((data["ENTRANCES_AVG"]))) / 2.0)) - (1.0)))), ((((data["cu__currency_3"]) - (data["ENTRANCES_MEDI"])))))))))/2.0)) 
    v["i106"] = np.tanh((((np.where(data["PRODUCT_COMBINATION_POS_industry_without_interest"]>0, np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), (((-1.0*((data["LIVINGAPARTMENTS_AVG"])))))), data["LIVINGAPARTMENTS_AVG"] )) + ((((data["CNT_INSTALMENT_FUTURE"]) > (3.0))*1.)))/2.0)) 
    v["i107"] = np.tanh((((((data["FLAG_CONT_MOBILE"]) < (data["cc_bal_SK_DPD_DEF"]))*1.)) + (np.where(np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))>0, np.tanh((data["FLAG_WORK_PHONE"])), data["NAME_GOODS_CATEGORY_Direct_Sales"] )))) 
    v["i108"] = np.tanh(((((((data["FLAG_DOCUMENT_11"]) - (((np.maximum(((data["FLAG_DOCUMENT_11"])), ((data["SK_DPD"])))) / 2.0)))) * (((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * (3.0))))) * 2.0)) 
    v["i109"] = np.tanh(((((((data["NAME_TYPE_SUITE_Children"]) * (((((data["NAME_TYPE_SUITE_Children"]) * (np.tanh((data["ty__Microloan"]))))) - (np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Car_repairs"])), ((np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Car_repairs"])), ((-2.0))))))))))) * 2.0)) * 2.0)) 
    v["i110"] = np.tanh(np.minimum(((data["cnt_FLAG_DOCUMENT_13"])), ((((((((3.0) - (data["ty__Mortgage"]))) * 2.0)) + (((data["cnt_FLAG_DOCUMENT_15"]) + (data["cnt_FLAG_DOCUMENT_15"])))))))) 
    v["i111"] = np.tanh((((data["NAME_GOODS_CATEGORY_Insurance"]) + ((((((((data["PRODUCT_COMBINATION_POS_industry_without_interest"]) * 2.0)) + (((((10.99971294403076172)) < ((14.19601345062255859)))*1.)))/2.0)) * (np.minimum(((2.0)), (((((data["PRODUCT_COMBINATION_Card_X_Sell"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0))))))))/2.0)) 
    v["i112"] = np.tanh((((((((2.0) * (((((((-1.0*((data["avg_buro_MONTHS_BALANCE"])))) + (data["cu__currency_3"]))/2.0)) / 2.0)))) + ((((data["cnt_FLAG_DOCUMENT_11"]) + (((data["cc_bal_SK_DPD_DEF"]) / 2.0)))/2.0)))/2.0)) + (data["cu__currency_3"]))) 
    v["i113"] = np.tanh(((((((((0.0) < (data["DAYS_LAST_DUE_1ST_VERSION"]))*1.)) * 2.0)) < (data["DAYS_CREDIT_ENDDATE"]))*1.)) 
    v["i114"] = np.tanh(((((((-1.0*(((((np.tanh((data["NAME_CONTRACT_STATUS_Approved"]))) < (np.tanh((data["te_OCCUPATION_TYPE"]))))*1.))))) / 2.0)) + (data["cnt_FLAG_DOCUMENT_17"]))/2.0)) 
    v["i115"] = np.tanh(np.where(data["NAME_CLIENT_TYPE_Repeater"]>0, ((data["CODE_GENDER"]) * ((-1.0*(((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (((data["CODE_GENDER"]) * 2.0)))) + (data["te_ORGANIZATION_TYPE"]))/2.0)))))), data["NAME_GOODS_CATEGORY_Direct_Sales"] )) 
    v["i116"] = np.tanh(np.minimum(((((((-1.0*((data["DAYS_BIRTH"])))) + (2.0))/2.0))), ((((data["DAYS_BIRTH"]) * (((data["cnt_DAYS_BIRTH"]) + (np.tanh((2.0)))))))))) 
    v["i117"] = np.tanh((-1.0*(((((((-1.0) - (data["DAYS_ID_PUBLISH"]))) > ((((((-1.0) / 2.0)) + (np.tanh((data["cnt_FLAG_EMAIL"]))))/2.0)))*1.))))) 
    v["i118"] = np.tanh(np.where((-1.0*(((((data["FLAG_EMP_PHONE"]) + (data["DAYS_EMPLOYED"]))/2.0))))>0, -1.0, 1.0 )) 
    v["i119"] = np.tanh(((data["DAYS_BIRTH"]) * (np.maximum(((((-3.0) * (((((6.0)) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0))))), ((((1.0) - (np.maximum(((data["DAYS_BIRTH"])), ((data["AMT_CREDIT_y"]))))))))))) 
    v["i120"] = np.tanh(((data["CNT_FAM_MEMBERS"]) * (((((-1.0) + (data["DAYS_BIRTH"]))) / 2.0)))) 
    v["i121"] = np.tanh((((((0.0) + (np.where(data["NAME_EDUCATION_TYPE"]>0, data["AMT_CREDIT_x"], ((data["FLAG_WORK_PHONE"]) / 2.0) )))) + (((data["AMT_CREDIT_x"]) * ((-1.0*((data["FLAG_WORK_PHONE"])))))))/2.0)) 
    v["i122"] = np.tanh(np.where(np.maximum(((data["te_FLAG_OWN_REALTY"])), ((data["NAME_GOODS_CATEGORY_Insurance"])))>0, ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * 2.0), (((0.0) > (data["cnt_NAME_CONTRACT_TYPE"]))*1.) )) 
    v["i123"] = np.tanh(((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + ((((data["MONTHS_BALANCE"]) > (np.maximum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((np.maximum(((0.0)), ((((2.0) + (data["WALLSMATERIAL_MODE"]))))))))))*1.)))) * 2.0)) * 2.0)) 
    v["i124"] = np.tanh(((((data["inst_AMT_PAYMENT"]) - (-1.0))) * (((data["MONTHS_BALANCE"]) * ((-1.0*(((((data["inst_DAYS_ENTRY_PAYMENT"]) < (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["NAME_CASH_LOAN_PURPOSE_Furniture"], data["inst_DAYS_INSTALMENT"] )))*1.))))))))) 
    v["i125"] = np.tanh(np.where(((data["CNT_INSTALMENT_FUTURE"]) / 2.0)>0, (((data["NAME_GOODS_CATEGORY_Construction_Materials"]) < (data["NAME_EDUCATION_TYPE"]))*1.), ((data["NAME_EDUCATION_TYPE"]) * (((data["NAME_EDUCATION_TYPE"]) * (data["NAME_GOODS_CATEGORY_Construction_Materials"])))) )) 
    v["i126"] = np.tanh((-1.0*((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["AMT_GOODS_PRICE_x"], ((((((np.tanh((-1.0))) + (data["PRODUCT_COMBINATION_Cash_Street__low"]))/2.0)) > (((((9.59368038177490234)) > (-1.0))*1.)))*1.) )) * 2.0))))) 
    v["i127"] = np.tanh(((((np.tanh((-3.0))) - (data["NAME_SELLER_INDUSTRY_Connectivity"]))) * (np.tanh((((data["NONLIVINGAREA_MODE"]) / 2.0)))))) 
    v["i128"] = np.tanh(((data["CODE_GENDER"]) * (np.minimum(((np.minimum(((data["NONLIVINGAPARTMENTS_MODE"])), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))), ((np.tanh((np.minimum(((data["NONLIVINGAPARTMENTS_MODE"])), ((((((data["NONLIVINGAPARTMENTS_MODE"]) * (data["WALLSMATERIAL_MODE"]))) * 2.0)))))))))))) 
    v["i129"] = np.tanh(((np.where(data["NAME_CASH_LOAN_PURPOSE_Furniture"]>0, ((((1.0) + (data["cnt_HOUR_APPR_PROCESS_START_x"]))) + (data["NAME_GOODS_CATEGORY_Insurance"])), np.minimum(((0.0)), ((((((data["WALLSMATERIAL_MODE"]) * (-1.0))) / 2.0)))) )) / 2.0)) 
    v["i130"] = np.tanh(np.minimum(((((data["cnt_FLAG_DOCUMENT_16"]) - ((((((((0.0) * 2.0)) / 2.0)) > ((-1.0*((data["NAME_GOODS_CATEGORY_House_Construction"])))))*1.))))), (((-1.0*(((((data["FLAG_DOCUMENT_16"]) + (data["ty__Another_type_of_loan"]))/2.0)))))))) 
    v["i131"] = np.tanh(np.where(((1.0) * ((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) > (data["NONLIVINGAPARTMENTS_MODE"]))*1.)))>0, ((data["cc_bal_SK_ID_PREV"]) / 2.0), np.maximum(((data["FLAG_DOCUMENT_21"])), ((data["NAME_SELLER_INDUSTRY_Auto_technology"]))) )) 
    v["i132"] = np.tanh(((data["AMT_CREDIT_x"]) * (np.maximum(((-3.0)), ((((((-1.0*((data["DAYS_ID_PUBLISH"])))) + (((((((-2.0) > (data["DAYS_ID_PUBLISH"]))*1.)) < (data["NONLIVINGAPARTMENTS_MODE"]))*1.)))/2.0))))))) 
    v["i133"] = np.tanh(((((np.minimum(((((2.0) - (data["NONLIVINGAPARTMENTS_MODE"])))), ((data["NONLIVINGAPARTMENTS_MODE"])))) * (((data["te_LIVE_CITY_NOT_WORK_CITY"]) - (data["NONLIVINGAPARTMENTS_MODE"]))))) * 2.0)) 
    v["i134"] = np.tanh((((((-1.0*(((((data["cc_bal_cc_bal_status__Sent_proposal"]) + (((((2.0) + ((5.66302442550659180)))) * (data["AMT_CREDIT_x"]))))/2.0))))) * (((data["cc_bal_cc_bal_status__Sent_proposal"]) / 2.0)))) - (data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]))) 
    v["i135"] = np.tanh(((data["NAME_CASH_LOAN_PURPOSE_Medicine"]) - (np.where(data["NAME_CASH_LOAN_PURPOSE_Medicine"]>0, data["Active"], ((((((np.tanh(((5.0)))) > (data["Active"]))*1.)) < (np.tanh((data["inst_AMT_PAYMENT"]))))*1.) )))) 
    v["i136"] = np.tanh(((2.0) * (((((np.maximum(((((data["AMT_CREDIT_MAX_OVERDUE"]) + (data["cnt_FLAG_DOCUMENT_2"])))), ((((data["AMT_CREDIT_MAX_OVERDUE"]) * ((9.91402626037597656))))))) * ((9.91402626037597656)))) * 2.0)))) 
    v["i137"] = np.tanh(np.where(data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]>0, data["SK_ID_BUREAU"], np.where(((((((1.0) + (2.0))/2.0)) < (data["SK_ID_BUREAU"]))*1.)>0, data["SK_ID_PREV_x"], ((data["SK_ID_BUREAU"]) * (data["SK_ID_PREV_x"])) ) )) 
    v["i138"] = np.tanh(np.where(np.minimum(((1.0)), ((((((-1.0*((-3.0)))) > (data["avg_buro_buro_count"]))*1.))))>0, np.minimum(((((data["NAME_YIELD_GROUP_high"]) * (data["avg_buro_buro_count"])))), ((((data["PRODUCT_COMBINATION_Cash"]) / 2.0)))), 1.0 )) 
    v["i139"] = np.tanh((((((((data["AMT_CREDIT_SUM"]) * 2.0)) < ((((1.0) + (np.where((-1.0*(((6.0))))>0, data["avg_buro_buro_count"], ((((-1.0) / 2.0)) - (2.0)) )))/2.0)))*1.)) * 2.0)) 
    v["i140"] = np.tanh(((((((-1.0*((-2.0)))) < (((data["NAME_GOODS_CATEGORY_Insurance"]) + (data["SK_ID_BUREAU"]))))*1.)) - (np.where(((data["avg_buro_buro_count"]) * (-1.0))>0, data["PRODUCT_COMBINATION_Cash_X_Sell__low"], (0.0) )))) 
    v["i141"] = np.tanh(np.minimum((((((((2.0) < (data["CNT_INSTALMENT_FUTURE"]))*1.)) / 2.0))), (((((data["cnt_FLAG_EMAIL"]) + (np.maximum(((3.0)), ((data["CNT_INSTALMENT_FUTURE"])))))/2.0))))) 
    v["i142"] = np.tanh(((((((((((np.maximum(((1.0)), (((((-3.0) > ((12.31715393066406250)))*1.))))) < (data["CNT_INSTALMENT"]))*1.)) * 2.0)) < (np.where(data["DAYS_BIRTH"]>0, data["EXT_SOURCE_3"], data["AMT_CREDIT_x"] )))*1.)) / 2.0)) 
    v["i143"] = np.tanh(np.minimum(((((data["NAME_GOODS_CATEGORY_Insurance"]) - ((((data["NAME_GOODS_CATEGORY_Insurance"]) + (data["te_WALLSMATERIAL_MODE"]))/2.0))))), ((((((-1.0) * ((((data["NAME_CLIENT_TYPE_Repeater"]) < (data["DAYS_BIRTH"]))*1.)))) / 2.0))))) 
    v["i144"] = np.tanh((((((((data["SK_ID_PREV_y"]) * (data["inst_AMT_PAYMENT"]))) * 2.0)) + (((data["SK_ID_PREV_y"]) * (np.minimum(((data["EXT_SOURCE_2"])), (((((data["inst_AMT_PAYMENT"]) > (0.0))*1.))))))))/2.0)) 
    v["i145"] = np.tanh(((((data["AMT_GOODS_PRICE_x"]) * (np.tanh((np.minimum((((-1.0*((data["CNT_INSTALMENT_FUTURE"]))))), (((((((data["NAME_CLIENT_TYPE_Repeater"]) + ((-1.0*((data["AMT_GOODS_PRICE_x"])))))/2.0)) / 2.0))))))))) / 2.0)) 
    v["i146"] = np.tanh(np.tanh((((((((((data["NAME_CLIENT_TYPE_Repeater"]) + (data["PRODUCT_COMBINATION_Cash_X_Sell__high"]))/2.0)) + (np.where((((((data["AMT_CREDIT_x"]) / 2.0)) > (data["AMT_CREDIT_y"]))*1.)>0, data["AMT_CREDIT_x"], data["NAME_GOODS_CATEGORY_Insurance"] )))/2.0)) / 2.0)))) 
    v["i147"] = np.tanh(((((-1.0*(((((((data["DAYS_LAST_DUE_1ST_VERSION"]) > (((((2.0)) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)))*1.)) * (data["inst_NUM_INSTALMENT_NUMBER"])))))) + ((((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) < (data["inst_NUM_INSTALMENT_NUMBER"]))*1.)) * (data["NAME_CONTRACT_STATUS_Approved"]))))/2.0)) 
    v["i148"] = np.tanh(np.where(((data["AMT_GOODS_PRICE_x"]) * 2.0)>0, data["Completed"], (((((np.tanh((data["SK_ID_PREV_x"]))) + (np.tanh((1.0))))/2.0)) / 2.0) )) 
    v["i149"] = np.tanh(((3.0) * (np.minimum(((np.minimum(((((data["AMT_CREDIT_x"]) + (1.0)))), ((((data["NAME_GOODS_CATEGORY_Insurance"]) * (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))), (((((data["AMT_CREDIT_x"]) + (1.0))/2.0))))))) 
    v["i150"] = np.tanh(((data["CNT_INSTALMENT_FUTURE"]) * ((((((((data["AMT_GOODS_PRICE_x"]) * ((-1.0*((data["SK_DPD"])))))) < (data["SK_DPD"]))*1.)) * 2.0)))) 
    v["i151"] = np.tanh(((((((-1.0*((np.tanh((np.minimum(((data["AMT_GOODS_PRICE_x"])), ((((data["FLAG_DOCUMENT_2"]) * (data["SK_DPD"]))))))))))) + (((((data["SK_DPD"]) - (((data["AMT_GOODS_PRICE_x"]) / 2.0)))) / 2.0)))/2.0)) / 2.0)) 
    v["i152"] = np.tanh(np.where(data["SK_DPD"]>0, (-1.0*((data["te_OCCUPATION_TYPE"]))), ((((np.where(data["AMT_CREDIT_x"]>0, (-1.0*((data["te_OCCUPATION_TYPE"]))), ((((-2.0) / 2.0)) / 2.0) )) / 2.0)) / 2.0) )) 
    v["i153"] = np.tanh((((((data["AMT_CREDIT_x"]) > (np.maximum((((-1.0*((data["cnt_NAME_EDUCATION_TYPE"]))))), ((((((((-1.0*((np.minimum(((data["AMT_CREDIT_x"])), ((np.tanh((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))))) / 2.0)) > (data["SK_DPD"]))*1.))))))*1.)) / 2.0)) 
    v["i154"] = np.tanh(np.minimum((((((data["cnt_FLAG_DOCUMENT_17"]) + (data["cnt_FLAG_DOCUMENT_15"]))/2.0))), ((np.tanh((((np.maximum(((data["cnt_FLAG_DOCUMENT_17"])), (((-1.0*((data["AMT_GOODS_PRICE_x"]))))))) - (np.where(-3.0>0, data["cnt_FLAG_DOCUMENT_17"], data["cnt_NAME_HOUSING_TYPE"] ))))))))) 
    v["i155"] = np.tanh(((data["NAME_SELLER_INDUSTRY_MLM_partners"]) * (((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], (-1.0*((data["NAME_GOODS_CATEGORY_Direct_Sales"]))) )) - (np.maximum(((np.maximum(((3.0)), ((data["SK_DPD"]))))), ((0.0)))))))) 
    v["i156"] = np.tanh(np.minimum(((((np.minimum(((data["FLAG_DOCUMENT_21"])), ((((data["AMT_CREDIT_x"]) / 2.0))))) * (data["AMT_REQ_CREDIT_BUREAU_MON"])))), ((np.tanh((((np.minimum(((((data["AMT_CREDIT_x"]) / 2.0))), ((0.0)))) / 2.0))))))) 
    v["i157"] = np.tanh(((((((((((data["AMT_GOODS_PRICE_x"]) < (((np.tanh(((10.0)))) / 2.0)))*1.)) > (((((((data["inst_AMT_PAYMENT"]) + (0.0))/2.0)) > ((-1.0*((0.0)))))*1.)))*1.)) / 2.0)) / 2.0)) 
    v["i158"] = np.tanh((((((((3.0) < ((((((data["NAME_YIELD_GROUP_high"]) + ((((-2.0) > (np.tanh((data["FLAG_DOCUMENT_21"]))))*1.)))/2.0)) + (data["AMT_CREDIT_y"]))))*1.)) - (data["cnt_FLAG_DOCUMENT_21"]))) * 2.0)) 
    v["i159"] = np.tanh(((((((data["FLAG_DOCUMENT_2"]) + ((-1.0*(((((data["AMT_GOODS_PRICE_x"]) > (np.tanh((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((data["AMT_GOODS_PRICE_x"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"])))))))))*1.))))))) / 2.0)) / 2.0)) 
    v["i160"] = np.tanh(((((((data["AMT_CREDIT_x"]) + (data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]))/2.0)) + ((((-3.0) > (((data["AMT_CREDIT_x"]) + (((data["AMT_CREDIT_x"]) * 2.0)))))*1.)))/2.0)) 
    v["i161"] = np.tanh(((data["AMT_GOODS_PRICE_x"]) * (np.where(((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))>0, -1.0, ((data["AMT_GOODS_PRICE_x"]) * (np.tanh((data["CODE_REJECT_REASON_VERIF"])))) )))) 
    v["i162"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Furniture"]>0, ((data["OCCUPATION_TYPE"]) / 2.0), ((((((data["te_WEEKDAY_APPR_PROCESS_START"]) / 2.0)) / 2.0)) / 2.0) )) 
    v["i163"] = np.tanh((-1.0*((((3.0) * ((((data["CODE_REJECT_REASON_VERIF"]) > ((((9.0)) + ((-1.0*((np.maximum(((np.minimum(((data["te_REG_REGION_NOT_LIVE_REGION"])), ((data["te_REG_REGION_NOT_LIVE_REGION"]))))), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))))))*1.))))))) 
    v["i164"] = np.tanh((((data["cnt_FLAG_DOCUMENT_17"]) > (((data["CODE_REJECT_REASON_VERIF"]) - (np.where(data["SK_DPD"]>0, -3.0, ((data["SK_DPD"]) * 2.0) )))))*1.)) 
    v["i165"] = np.tanh(((((data["NAME_EDUCATION_TYPE"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) + (((((np.minimum(((data["cnt_FLAG_DOCUMENT_17"])), (((((((10.18762969970703125)) - (data["CODE_REJECT_REASON_VERIF"]))) + ((3.18121862411499023))))))) / 2.0)) / 2.0)))) 
    v["i166"] = np.tanh((((((-1.0*((data["NAME_YIELD_GROUP_high"])))) * (((((-1.0*((-1.0)))) < (((data["cnt_HOUR_APPR_PROCESS_START_x"]) * (2.0))))*1.)))) / 2.0)) 
    v["i167"] = np.tanh((((-1.0*((((np.minimum(((((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * (data["PRODUCT_COMBINATION_Cash_Street__high"]))) * 2.0))), ((2.0)))) + ((((3.0) < (data["PRODUCT_COMBINATION_Cash_X_Sell__low"]))*1.))))))) * 2.0)) 
    v["i168"] = np.tanh(((data["SK_DPD"]) * (((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) + (((((data["PRODUCT_COMBINATION_Card_Street"]) + (((data["PRODUCT_COMBINATION_Card_Street"]) - (((data["NAME_SELLER_INDUSTRY_Connectivity"]) * 2.0)))))) - (data["NAME_GOODS_CATEGORY_Insurance"]))))))) 
    v["i169"] = np.tanh(((data["AMT_CREDIT_x"]) * (((((((data["ty__Mortgage"]) > ((((data["NAME_GOODS_CATEGORY_Insurance"]) + (np.where(data["ty__Mortgage"]>0, 3.0, 1.0 )))/2.0)))*1.)) < (((data["ty__Mortgage"]) / 2.0)))*1.)))) 
    v["i170"] = np.tanh((-1.0*((np.where(data["ty__Mortgage"]>0, data["cc_bal_CNT_DRAWINGS_CURRENT"], (((((3.0) < ((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) + (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))/2.0)))*1.)) * ((-1.0*((data["cc_bal_CNT_DRAWINGS_CURRENT"]))))) ))))) 
    v["i171"] = np.tanh(np.where(0.0>0, -3.0, np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, (((data["DEF_30_CNT_SOCIAL_CIRCLE"]) > (data["NAME_GOODS_CATEGORY_Direct_Sales"]))*1.), np.where(data["cu__currency_3"]>0, 2.0, (((data["DEF_30_CNT_SOCIAL_CIRCLE"]) > (2.0))*1.) ) ) )) 
    v["i172"] = np.tanh(np.minimum((((-1.0*((np.where(((data["NAME_YIELD_GROUP_high"]) / 2.0)>0, data["ty__Mortgage"], data["NAME_GOODS_CATEGORY_Weapon"] )))))), ((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["cnt_HOUR_APPR_PROCESS_START_x"], data["NAME_GOODS_CATEGORY_Direct_Sales"] ))))) 
    v["i173"] = np.tanh((-1.0*((((((data["CHANNEL_TYPE_Car_dealer"]) + (((np.maximum(((data["CHANNEL_TYPE_Car_dealer"])), ((np.tanh((-2.0)))))) + (data["NAME_GOODS_CATEGORY_Fitness"]))))) + (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["NONLIVINGAPARTMENTS_MODE"], data["CHANNEL_TYPE_Car_dealer"] ))))))) 
    v["i174"] = np.tanh(np.where(np.where(((-2.0) + (data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]))>0, data["WEEKDAY_APPR_PROCESS_START_MONDAY"], data["NONLIVINGAPARTMENTS_MODE"] )>0, data["NAME_GOODS_CATEGORY_Computers"], np.minimum(((((data["NONLIVINGAPARTMENTS_MODE"]) / 2.0))), ((((data["NAME_GOODS_CATEGORY_Computers"]) * (data["NAME_GOODS_CATEGORY_Gardening"]))))) )) 
    v["i175"] = np.tanh((((-1.0*((np.where(data["NAME_YIELD_GROUP_high"]>0, data["NONLIVINGAPARTMENTS_MODE"], (-1.0*((np.where((12.60592746734619141)>0, data["NONLIVINGAPARTMENTS_MODE"], ((data["NONLIVINGAPARTMENTS_MODE"]) * 2.0) )))) ))))) * 2.0)) 
    v["i176"] = np.tanh((-1.0*((((np.tanh(((((((data["FLAG_DOCUMENT_17"]) < (data["WEEKDAY_APPR_PROCESS_START_MONDAY"]))*1.)) - (data["NAME_GOODS_CATEGORY_Weapon"]))))) * (np.where(data["WEEKDAY_APPR_PROCESS_START_MONDAY"]>0, ((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) + (data["AMT_REQ_CREDIT_BUREAU_WEEK"])), data["FLAG_DOCUMENT_17"] ))))))) 
    v["i177"] = np.tanh(np.where(((((10.0)) < (data["NONLIVINGAPARTMENTS_MODE"]))*1.)>0, 3.0, np.where(data["NONLIVINGAPARTMENTS_MODE"]>0, ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * 2.0), data["AMT_REQ_CREDIT_BUREAU_DAY"] ) )) 
    v["i178"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["avg_buro_buro_count"], np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((data["NAME_YIELD_GROUP_high"]) * (np.tanh((((data["avg_buro_buro_count"]) * 2.0)))))))) )) 
    v["i179"] = np.tanh(((((-1.0*((((-1.0) - (data["avg_buro_buro_count"])))))) < (np.minimum(((data["te_HOUR_APPR_PROCESS_START_x"])), ((((data["DAYS_ENDDATE_FACT"]) * 2.0))))))*1.)) 
    v["i180"] = np.tanh(np.minimum((((((3.0) < (data["AMT_CREDIT_x"]))*1.))), ((((((((3.0) < (data["AMT_CREDIT_x"]))*1.)) + (np.maximum(((data["AMT_CREDIT_x"])), ((data["cnt_HOUR_APPR_PROCESS_START_x"])))))/2.0))))) 
    v["i181"] = np.tanh((-1.0*(((((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["NAME_GOODS_CATEGORY_Insurance"], data["NAME_GOODS_CATEGORY_Direct_Sales"] )) > (((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * (data["AMT_GOODS_PRICE_x"]))) * 2.0)))*1.)) / 2.0))))) 
    v["i182"] = np.tanh((((((data["FLAG_CONT_MOBILE"]) + (np.tanh((np.tanh((np.tanh((np.tanh((data["CODE_GENDER"]))))))))))/2.0)) * (data["AMT_CREDIT_x"]))) 
    v["i183"] = np.tanh(((((((((((((np.minimum(((data["cc_bal_SK_DPD_DEF"])), ((0.0)))) * 2.0)) < (0.0))*1.)) < ((-1.0*((data["AMT_GOODS_PRICE_x"])))))*1.)) / 2.0)) / 2.0)) 
    v["i184"] = np.tanh(((data["NAME_YIELD_GROUP_high"]) * ((((data["cc_bal_SK_DPD_DEF"]) < (((((-1.0) + (data["inst_NUM_INSTALMENT_VERSION"]))) + (((((0.0)) < (((data["NAME_GOODS_CATEGORY_Insurance"]) / 2.0)))*1.)))))*1.)))) 
    v["i185"] = np.tanh((((np.tanh(((((((data["cc_bal_SK_DPD_DEF"]) - (data["AMT_DOWN_PAYMENT"]))) < (np.where((((5.78417444229125977)) + ((-1.0*((data["cc_bal_SK_DPD_DEF"])))))>0, data["NAME_GOODS_CATEGORY_Insurance"], data["NAME_GOODS_CATEGORY_Insurance"] )))*1.)))) < (data["AMT_DOWN_PAYMENT"]))*1.)) 
    v["i186"] = np.tanh((((np.tanh((((data["AMT_CREDIT_x"]) / 2.0)))) + (np.tanh((((data["RATE_DOWN_PAYMENT"]) * ((((-1.0*(((((data["CNT_INSTALMENT_FUTURE"]) < (((data["AMT_CREDIT_x"]) * 2.0)))*1.))))) / 2.0)))))))/2.0)) 
    v["i187"] = np.tanh(np.minimum(((((((((((data["AMT_ANNUITY_x"]) > (data["AMT_GOODS_PRICE_x"]))*1.)) / 2.0)) + (np.minimum(((data["NAME_PORTFOLIO_Cards"])), ((data["NAME_GOODS_CATEGORY_Insurance"])))))/2.0))), (((6.0))))) 
    v["i188"] = np.tanh((((((((np.minimum(((data["AMT_GOODS_PRICE_x"])), (((((8.0)) + (data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"])))))) + ((-1.0*((3.0)))))/2.0)) * (((((-1.0*((data["AMT_GOODS_PRICE_x"])))) < ((0.21274571120738983)))*1.)))) / 2.0)) 
    v["i189"] = np.tanh(((((((np.maximum((((((((((((data["SK_DPD"]) + (data["SK_DPD"]))/2.0)) < (data["cc_bal_SK_ID_PREV"]))*1.)) + ((((data["CNT_INSTALMENT_FUTURE"]) + (data["SK_DPD"]))/2.0)))/2.0))), ((data["NAME_GOODS_CATEGORY_Insurance"])))) / 2.0)) * 2.0)) / 2.0)) 
    v["i190"] = np.tanh(((((-1.0) / 2.0)) * (np.where(data["CODE_REJECT_REASON_HC"]>0, data["NFLAG_INSURED_ON_APPROVAL"], np.minimum(((np.minimum(((data["MONTHS_BALANCE"])), ((2.0))))), ((0.0))) )))) 
    v["i191"] = np.tanh(((((((2.39028859138488770)) > (np.maximum(((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), (((-1.0*((((data["AMT_CREDIT_x"]) * 2.0)))))))))*1.)) + (-1.0))) 
    v["i192"] = np.tanh(np.where(np.tanh((data["NAME_YIELD_GROUP_high"]))>0, np.maximum((((((np.tanh(((((data["CHANNEL_TYPE_Car_dealer"]) + (data["DAYS_DECISION"]))/2.0)))) + (data["CHANNEL_TYPE_Car_dealer"]))/2.0))), ((data["SK_ID_PREV_x"]))), (-1.0*((data["CHANNEL_TYPE_Car_dealer"]))) )) 
    v["i193"] = np.tanh(np.where(data["SK_ID_PREV_y"]>0, ((np.minimum(((np.minimum(((data["NAME_YIELD_GROUP_high"])), ((((data["NAME_YIELD_GROUP_high"]) + (3.0))))))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) * (data["NAME_CONTRACT_STATUS_Refused"])), data["NAME_GOODS_CATEGORY_Direct_Sales"] )) 
    v["i194"] = np.tanh(np.minimum(((1.0)), (((-1.0*((np.maximum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((((data["DAYS_BIRTH"]) * (np.tanh((data["AMT_REQ_CREDIT_BUREAU_YEAR"])))))))))))))) 
    v["i195"] = np.tanh((((((np.minimum(((data["CNT_INSTALMENT"])), ((((((-1.0) * 2.0)) * (((data["CNT_INSTALMENT"]) / 2.0))))))) > (data["SK_ID_PREV_y"]))*1.)) / 2.0)) 
    v["i196"] = np.tanh((-1.0*(((((data["DAYS_BIRTH"]) > (((3.0) - ((((((data["DAYS_BIRTH"]) + (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (-3.0))))/2.0)) * 2.0)))))*1.))))) 
    v["i197"] = np.tanh(((((((((((data["cnt_DAYS_BIRTH"]) / 2.0)) + ((3.49354934692382812)))/2.0)) < (data["DAYS_ID_PUBLISH"]))*1.)) * (data["DAYS_ID_PUBLISH"]))) 
    v["i198"] = np.tanh((((np.maximum(((((data["NAME_INCOME_TYPE"]) + (-2.0)))), (((((data["cnt_DAYS_BIRTH"]) > ((((data["NAME_INCOME_TYPE"]) < (0.0))*1.)))*1.))))) < (data["NAME_GOODS_CATEGORY_House_Construction"]))*1.)) 
    v["i199"] = np.tanh(((np.minimum((((7.0))), ((((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((np.tanh((((np.tanh((((data["cnt_FLAG_EMP_PHONE"]) + (data["DAYS_EMPLOYED"]))))) * ((9.89645957946777344))))))))) * 2.0))))) * 2.0)) 
    v["i200"] = np.tanh(((np.tanh(((((data["SK_ID_BUREAU"]) < (-1.0))*1.)))) * ((((data["NAME_YIELD_GROUP_high"]) > (((data["SK_ID_BUREAU"]) - (np.maximum(((data["te_FLAG_DOCUMENT_20"])), ((((-3.0) * 2.0))))))))*1.)))) 
    v["i201"] = np.tanh(((np.minimum(((((data["NAME_CLIENT_TYPE_New"]) * (((data["DAYS_LAST_PHONE_CHANGE"]) * (data["SK_ID_BUREAU"])))))), ((((((((3.0) < (data["NAME_GOODS_CATEGORY_Direct_Sales"]))*1.)) > (data["DAYS_LAST_PHONE_CHANGE"]))*1.))))) / 2.0)) 
    v["i202"] = np.tanh((-1.0*((((np.maximum((((((-2.0) > (data["DAYS_BIRTH"]))*1.))), ((((np.where(((3.0) - (data["te_LIVE_REGION_NOT_WORK_REGION"]))>0, -3.0, data["DAYS_BIRTH"] )) * 2.0))))) * 2.0))))) 
    v["i203"] = np.tanh((-1.0*((np.where(np.minimum(((1.0)), ((((data["SK_ID_BUREAU"]) + (-2.0)))))>0, -1.0, ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_home"]) * 2.0) ))))) 
    v["i204"] = np.tanh(np.minimum((((-1.0*((np.where(data["te_FLAG_DOCUMENT_20"]>0, data["NAME_GOODS_CATEGORY_Weapon"], data["NAME_GOODS_CATEGORY_Direct_Sales"] )))))), ((((((((data["NAME_GOODS_CATEGORY_Weapon"]) + (((data["NAME_CONTRACT_STATUS_Approved"]) * (data["NFLAG_LAST_APPL_IN_DAY"]))))) * 2.0)) / 2.0))))) 
    v["i205"] = np.tanh(((((data["te_FLAG_DOCUMENT_20"]) * (np.where(data["te_FLAG_DOCUMENT_20"]>0, data["te_FLAG_DOCUMENT_20"], (6.34485292434692383) )))) * (np.where(np.tanh(((4.0)))>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], data["NAME_GOODS_CATEGORY_Insurance"] )))) 
    v["i206"] = np.tanh(((data["SK_ID_PREV_x"]) * (((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) - ((-1.0*(((((((data["SK_ID_PREV_x"]) * 2.0)) > ((14.96901893615722656)))*1.))))))) - (data["NAME_GOODS_CATEGORY_Direct_Sales"]))))) 
    v["i207"] = np.tanh(((data["cnt_FLAG_DOCUMENT_10"]) + (((((np.where(data["cnt_FLAG_DOCUMENT_17"]>0, data["NAME_YIELD_GROUP_high"], data["NAME_GOODS_CATEGORY_Insurance"] )) * (data["FLAG_DOCUMENT_17"]))) * ((8.24418640136718750)))))) 
    v["i208"] = np.tanh(np.where(((-1.0) + (data["CNT_INSTALMENT"]))>0, data["REG_REGION_NOT_WORK_REGION"], (-1.0*((data["NAME_GOODS_CATEGORY_Medical_Supplies"]))) )) 
    v["i209"] = np.tanh(np.where(((data["AMT_ANNUITY_x"]) * (data["FLAG_DOCUMENT_15"]))>0, np.minimum((((0.83881276845932007))), ((((-1.0) - (data["AMT_ANNUITY_x"]))))), ((data["FLAG_DOCUMENT_15"]) * (data["AMT_ANNUITY_x"])) )) 
    v["i210"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["te_FLAG_DOCUMENT_3"], (((-1.0*((((0.0) - ((((data["AMT_GOODS_PRICE_x"]) < (((data["AMT_ANNUITY_x"]) - (((((9.0)) < (3.0))*1.)))))*1.))))))) / 2.0) )) 
    v["i211"] = np.tanh(((((-2.0) / 2.0)) + ((((data["AMT_CREDIT_x"]) > (((np.where(((3.0) * (data["AMT_ANNUITY_x"]))>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], data["AMT_ANNUITY_x"] )) * 2.0)))*1.)))) 
    v["i212"] = np.tanh((-1.0*((((((((((2.0) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) - (np.minimum(((-2.0)), (((12.27902317047119141))))))) < (((data["AMT_GOODS_PRICE_x"]) + (data["AMT_ANNUITY_x"]))))*1.))))) 
    v["i213"] = np.tanh((((((-3.0) > (((np.where((((((((data["AMT_CREDIT_x"]) / 2.0)) + (data["CNT_PAYMENT"]))/2.0)) / 2.0)>0, data["DAYS_BIRTH"], ((data["CNT_PAYMENT"]) / 2.0) )) * 2.0)))*1.)) * 2.0)) 
    v["i214"] = np.tanh(np.minimum(((((((data["AMT_CREDIT_x"]) / 2.0)) / 2.0))), (((((((-1.0) * 2.0)) > (((data["AMT_CREDIT_x"]) - ((3.90542602539062500)))))*1.))))) 
    v["i215"] = np.tanh(((((np.tanh(((-1.0*((np.minimum(((data["AMT_GOODS_PRICE_x"])), ((((data["CHANNEL_TYPE_AP___Cash_loan_"]) / 2.0)))))))))) / 2.0)) + ((((data["CHANNEL_TYPE_AP___Cash_loan_"]) > ((((4.0)) + (np.tanh((data["NAME_GOODS_CATEGORY_Direct_Sales"]))))))*1.)))) 
    v["i216"] = np.tanh((((-1.0*(((((((np.minimum(((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * 2.0))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) * ((6.0)))) > (data["HOUR_APPR_PROCESS_START_x"]))*1.))))) / 2.0)) 
    v["i217"] = np.tanh((((np.where(data["FLAG_DOCUMENT_2"]>0, (((-1.0) < (-2.0))*1.), ((((-1.0*((data["cnt_FLAG_DOCUMENT_2"])))) + (3.0))/2.0) )) < ((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["te_HOUR_APPR_PROCESS_START_x"]))/2.0)))*1.)) 
    v["i218"] = np.tanh(((3.0) * (np.tanh((np.maximum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((np.minimum(((((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) > ((((data["cnt_REGION_RATING_CLIENT"]) < (data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]))*1.)))*1.)) > (data["cnt_FLAG_DOCUMENT_2"]))*1.))), ((data["cnt_HOUR_APPR_PROCESS_START_x"]))))))))))) 
    v["i219"] = np.tanh(((data["te_FLAG_DOCUMENT_18"]) * (((data["cnt_REGION_RATING_CLIENT_W_CITY"]) + (data["AMT_CREDIT_x"]))))) 
    v["i220"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, (-1.0*((((((((-1.0*((-1.0)))) * (((((((3.0)) + ((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))))/2.0)) / 2.0)))) + (data["cnt_FLAG_DOCUMENT_18"]))/2.0)))), data["cnt_FLAG_DOCUMENT_18"] )) 
    v["i221"] = np.tanh(((data["cnt_FLAG_DOCUMENT_18"]) - (np.tanh(((((0.0) > (np.maximum(((((((((data["REGION_POPULATION_RELATIVE"]) + (data["te_FLAG_DOCUMENT_18"]))) + (data["te_FLAG_DOCUMENT_18"]))) - (data["cc_bal_SK_DPD_DEF"])))), ((data["cc_bal_SK_DPD_DEF"])))))*1.)))))) 
    v["i222"] = np.tanh(((data["inst_NUM_INSTALMENT_VERSION"]) * (((data["SK_ID_PREV_x"]) * ((-1.0*(((((((((((((data["NAME_CLIENT_TYPE_Repeater"]) < (((-1.0) / 2.0)))*1.)) * 2.0)) > (data["AMT_GOODS_PRICE_x"]))*1.)) > (data["NAME_PORTFOLIO_Cards"]))*1.))))))))) 
    v["i223"] = np.tanh((((-1.0) + ((((((data["AMT_GOODS_PRICE_x"]) * (data["NAME_CLIENT_TYPE_New"]))) < ((((2.0) + (((data["SK_ID_PREV_x"]) / 2.0)))/2.0)))*1.)))/2.0)) 
    v["i224"] = np.tanh(np.where(data["HOUR_APPR_PROCESS_START_y"]>0, (-1.0*((np.minimum(((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])), ((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) * 2.0))))))), data["NAME_GOODS_CATEGORY_Insurance"] )) 
    v["i225"] = np.tanh(((((data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"]) * (data["cc_bal_CNT_DRAWINGS_CURRENT"]))) * ((-1.0*((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, ((data["cc_bal_CNT_DRAWINGS_CURRENT"]) * (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, -2.0, data["inst_NUM_INSTALMENT_VERSION"] ))), data["inst_NUM_INSTALMENT_VERSION"] ))))))) 
    v["i226"] = np.tanh(((data["AMT_ANNUITY_x"]) * ((((np.minimum((((((((1.0) / 2.0)) < (((data["CNT_INSTALMENT_FUTURE"]) / 2.0)))*1.))), ((data["AMT_ANNUITY_x"])))) > (data["avg_buro_buro_bal_status_C"]))*1.)))) 
    v["i227"] = np.tanh((-1.0*(((((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) + ((((data["AMT_GOODS_PRICE_x"]) > (((((((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["NAME_GOODS_CATEGORY_Insurance"])))) > ((2.34558510780334473)))*1.)) + (data["AMT_ANNUITY_x"]))/2.0)))*1.)))/2.0))))) 
    v["i228"] = np.tanh(np.tanh((((((np.tanh(((((((data["avg_buro_buro_bal_status_C"]) + ((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) + (data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]))/2.0)))) > (np.tanh((3.0))))*1.)))) - (data["AMT_ANNUITY_x"]))) + (data["AMT_CREDIT_x"]))))) 
    v["i229"] = np.tanh(np.minimum(((((((data["NAME_GOODS_CATEGORY_Insurance"]) - ((((data["AMT_GOODS_PRICE_x"]) > (1.0))*1.)))) / 2.0))), (((-1.0*((((((-1.0*((-2.0)))) < (data["AMT_GOODS_PRICE_x"]))*1.)))))))) 
    v["i230"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, np.where(data["AMT_CREDIT_x"]>0, (2.79278469085693359), data["te_CODE_GENDER"] ), (((-2.0) > (((data["te_FLAG_DOCUMENT_8"]) / 2.0)))*1.) )) 
    v["i231"] = np.tanh((((((((((((3.0) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) > (data["AMT_CREDIT_x"]))*1.)) < (((-3.0) * (data["cc_bal_SK_DPD_DEF"]))))*1.)) - (data["CODE_REJECT_REASON_SYSTEM"]))) 
    v["i232"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, data["HOUR_APPR_PROCESS_START_y"], (((-1.0*(((((((np.maximum(((data["HOUR_APPR_PROCESS_START_y"])), ((data["CNT_INSTALMENT"])))) + (data["NAME_GOODS_CATEGORY_Weapon"]))/2.0)) / 2.0))))) / 2.0) )) 
    v["i233"] = np.tanh(((((data["NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans"]) * (data["FLAG_DOCUMENT_17"]))) + (np.maximum(((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans"])), (((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans"])))))))), (((((((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) / 2.0)) > ((6.61112833023071289)))*1.))))))) 
    v["i234"] = np.tanh(np.minimum((((((np.maximum(((data["CNT_INSTALMENT_FUTURE"])), ((data["cnt_FLAG_DOCUMENT_13"])))) > (3.0))*1.))), ((((data["AMT_CREDIT_x"]) - ((((((((data["cnt_FLAG_DOCUMENT_13"]) * 2.0)) > (data["CNT_INSTALMENT_FUTURE"]))*1.)) + (-2.0)))))))) 
    v["i235"] = np.tanh((-1.0*((np.where(data["NAME_GOODS_CATEGORY_Fitness"]>0, data["EXT_SOURCE_3"], (((np.minimum(((data["NAME_GOODS_CATEGORY_Fitness"])), ((np.tanh(((-1.0*((data["EXT_SOURCE_3"]))))))))) + (((data["cnt_FLAG_DOCUMENT_10"]) / 2.0)))/2.0) ))))) 
    v["i236"] = np.tanh(np.minimum(((((2.0) - (data["CNT_INSTALMENT"])))), ((np.tanh((np.where(data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]>0, data["NAME_CASH_LOAN_PURPOSE_Furniture"], np.where(data["cc_bal_SK_DPD_DEF"]>0, -3.0, ((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) * ((7.0))) ) ))))))) 
    v["i237"] = np.tanh(np.where(data["avg_buro_buro_count"]>0, np.where(data["cc_bal_SK_DPD_DEF"]>0, data["avg_buro_buro_count"], data["cnt_FLAG_DOCUMENT_13"] ), np.minimum(((data["cc_bal_SK_DPD_DEF"])), ((data["cnt_FLAG_DOCUMENT_13"]))) )) 
    v["i238"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_House_Construction"]>0, data["avg_buro_buro_count"], (((((-2.0) / 2.0)) + ((((data["NAME_GOODS_CATEGORY_House_Construction"]) < (((((data["avg_buro_buro_count"]) - (data["NAME_GOODS_CATEGORY_Fitness"]))) * (data["NAME_GOODS_CATEGORY_Fitness"]))))*1.)))/2.0) )) 
    v["i239"] = np.tanh(((((((((((data["cnt_FLAG_DOCUMENT_14"]) / 2.0)) + ((-1.0*((data["nans"])))))/2.0)) / 2.0)) + (((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) * (data["nans"]))))/2.0)) 
    v["i240"] = np.tanh(np.tanh(((((((((((data["CNT_INSTALMENT_FUTURE"]) * (data["CNT_INSTALMENT_FUTURE"]))) / 2.0)) > ((((data["REG_REGION_NOT_WORK_REGION"]) < (data["cc_bal_SK_DPD_DEF"]))*1.)))*1.)) / 2.0)))) 
    v["i241"] = np.tanh(((data["AMT_CREDIT_MAX_OVERDUE"]) * ((((12.82818222045898438)) + ((((((12.73882579803466797)) + ((((12.73882579803466797)) * ((13.13270282745361328)))))) * (((data["cc_bal_SK_DPD_DEF"]) * 2.0)))))))) 
    v["i242"] = np.tanh(((data["FLAG_LAST_APPL_PER_CONTRACT_N"]) * ((((((data["te_REG_REGION_NOT_LIVE_REGION"]) + (data["FONDKAPREMONT_MODE"]))/2.0)) + ((((data["FONDKAPREMONT_MODE"]) > ((((-1.0*((-3.0)))) / 2.0)))*1.)))))) 
    v["i243"] = np.tanh((-1.0*((np.tanh((((data["inst_AMT_PAYMENT"]) * (((data["inst_AMT_PAYMENT"]) * (np.maximum(((-1.0)), ((data["CNT_INSTALMENT"]))))))))))))) 
    v["i244"] = np.tanh(((np.where(((np.tanh((data["AMT_CREDIT_y"]))) + (((((1.0)) + (((data["cnt_FLAG_DOCUMENT_10"]) * 2.0)))/2.0)))>0, (((data["CNT_INSTALMENT_FUTURE"]) + (-1.0))/2.0), data["NAME_GOODS_CATEGORY_Insurance"] )) * (data["AMT_CREDIT_y"]))) 
    v["i245"] = np.tanh((((data["cnt_FLAG_DOCUMENT_10"]) + (np.minimum(((np.maximum(((data["AMT_CREDIT_y"])), ((data["CNT_PAYMENT"]))))), (((((data["cnt_FLAG_DOCUMENT_10"]) > (((((2.70514798164367676)) > (data["AMT_CREDIT_y"]))*1.)))*1.))))))/2.0)) 
    v["i246"] = np.tanh((((((((np.tanh((data["cnt_FLAG_DOCUMENT_3"]))) + (np.minimum((((((data["FLAG_DOCUMENT_3"]) < ((((data["cc_bal_SK_DPD_DEF"]) + (data["cnt_FLAG_DOCUMENT_3"]))/2.0)))*1.))), (((-1.0*((data["CNT_INSTALMENT"]))))))))/2.0)) / 2.0)) / 2.0)) 
    v["i247"] = np.tanh(np.maximum(((data["cnt_FLAG_DOCUMENT_5"])), ((((data["cc_bal_SK_DPD_DEF"]) + ((((((((data["cnt_FLAG_DOCUMENT_5"]) / 2.0)) / 2.0)) < (data["cc_bal_SK_DPD_DEF"]))*1.))))))) 
    v["i248"] = np.tanh((((((data["cc_bal_cc_bal_status__Refused"]) * 2.0)) + (((((data["NAME_SELLER_INDUSTRY_Construction"]) * ((4.0)))) * ((-1.0*(((((((data["NAME_SELLER_INDUSTRY_Construction"]) / 2.0)) > ((((7.86149215698242188)) / 2.0)))*1.))))))))/2.0)) 
    v["i249"] = np.tanh(((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) * (((((data["cc_bal_SK_DPD_DEF"]) * (data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]))) - ((-1.0*((data["NAME_GOODS_CATEGORY_Fitness"])))))))) 
    v["i250"] = np.tanh(((((((data["FLAG_DOCUMENT_17"]) * 2.0)) * (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))) - (((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * (((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) - (data["NAME_GOODS_CATEGORY_Weapon"]))))))) 
    v["i251"] = np.tanh(np.tanh((np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["avg_buro_buro_bal_status_2"], (-1.0*(((((((data["avg_buro_buro_bal_status_2"]) + (data["cc_bal_SK_DPD_DEF"]))) > (data["cnt_FLAG_DOCUMENT_10"]))*1.)))) )))) 
    v["i252"] = np.tanh(((((((10.22499179840087891)) < (data["NONLIVINGAPARTMENTS_MODE"]))*1.)) - ((((data["CHANNEL_TYPE_Car_dealer"]) + (((data["YEARS_BEGINEXPLUATATION_MODE"]) * 2.0)))/2.0)))) 
    v["i253"] = np.tanh(np.where(data["te_WALLSMATERIAL_MODE"]>0, data["YEARS_BUILD_AVG"], np.where(data["cnt_EMERGENCYSTATE_MODE"]>0, ((0.0) - (data["NAME_GOODS_CATEGORY_Direct_Sales"])), data["YEARS_BUILD_MEDI"] ) )) 
    v["i254"] = np.tanh((-1.0*((np.where(data["cc_bal_SK_DPD_DEF"]>0, data["NAME_GOODS_CATEGORY_Computers"], (((np.maximum(((-2.0)), (((1.0))))) < ((-1.0*(((((-1.0*((data["NAME_GOODS_CATEGORY_Computers"])))) + (2.0)))))))*1.) ))))) 
    v["i255"] = np.tanh(((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["avg_buro_buro_bal_status_5"], ((((data["DAYS_LAST_DUE_1ST_VERSION"]) * (data["cc_bal_SK_DPD_DEF"]))) * (data["DAYS_LAST_DUE_1ST_VERSION"])) )) * 2.0)) 
    v["i256"] = np.tanh((((np.minimum(((((np.tanh((np.minimum(((data["NFLAG_INSURED_ON_APPROVAL"])), (((12.13233852386474609))))))) + (data["NAME_GOODS_CATEGORY_Direct_Sales"])))), ((data["avg_buro_buro_bal_status_5"])))) > (np.where(data["avg_buro_buro_bal_status_5"]>0, data["cnt_FLAG_DOCUMENT_10"], data["cc_bal_SK_DPD_DEF"] )))*1.)) 
    v["i257"] = np.tanh(((data["NFLAG_INSURED_ON_APPROVAL"]) * ((((((((-1.0*((data["inst_SK_ID_PREV"])))) * (data["SK_ID_PREV_y"]))) / 2.0)) / 2.0)))) 
    v["i258"] = np.tanh(np.minimum((((((((((data["cc_bal_SK_DPD_DEF"]) + (data["SK_ID_PREV_x"]))) - (((((np.tanh((data["te_FLAG_EMAIL"]))) * (2.0))) / 2.0)))) > (2.0))*1.))), (((0.31483063101768494))))) 
    v["i259"] = np.tanh((-1.0*((np.minimum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((((((6.30967998504638672)) + ((((7.0)) - ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + ((((data["FLAG_DOCUMENT_10"]) > ((-1.0*((0.0)))))*1.)))/2.0)))))/2.0)))))))) 
    v["i260"] = np.tanh(np.minimum(((((((((-1.0*((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) / 2.0)) < (0.0))*1.))), (((((((((((0.0) > (data["cc_bal_SK_DPD_DEF"]))*1.)) * (data["cnt_EMERGENCYSTATE_MODE"]))) / 2.0)) * 2.0))))) 
    v["i261"] = np.tanh(((((((((data["NAME_GOODS_CATEGORY_Insurance"]) < (data["NAME_PORTFOLIO_Cards"]))*1.)) / 2.0)) + ((((((8.0)) / 2.0)) * ((((3.0) < (data["NAME_PORTFOLIO_Cards"]))*1.)))))/2.0)) 
    v["i262"] = np.tanh(np.where(data["FLAG_LAST_APPL_PER_CONTRACT_N"]>0, np.where((((data["DAYS_LAST_DUE_1ST_VERSION"]) < (((np.tanh((np.tanh((data["FLAG_LAST_APPL_PER_CONTRACT_N"]))))) / 2.0)))*1.)>0, data["DAYS_LAST_DUE_1ST_VERSION"], -3.0 ), (-1.0*((data["NAME_GOODS_CATEGORY_Education"]))) )) 
    v["i263"] = np.tanh(np.minimum(((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (((((((data["ty__Another_type_of_loan"]) + (data["DAYS_LAST_DUE_1ST_VERSION"]))) - (data["cc_bal_cc_bal_status__Refused"]))) + (data["DAYS_LAST_DUE_1ST_VERSION"]))))) * 2.0))), (((((10.71418285369873047)) - (data["ty__Another_type_of_loan"])))))) 
    v["i264"] = np.tanh((((((((-1.0) * 2.0)) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) > (np.maximum(((data["DAYS_FIRST_DRAWING"])), ((((data["CNT_INSTALMENT_FUTURE"]) * (np.minimum(((3.0)), ((data["inst_AMT_PAYMENT"]))))))))))*1.)) 
    v["i265"] = np.tanh(np.minimum((((((-2.0) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.))), (((((((((data["NAME_INCOME_TYPE"]) + (((-1.0) + (data["DAYS_BIRTH"]))))) + (2.0))) + (data["AMT_CREDIT_x"]))/2.0))))) 
    v["i266"] = np.tanh(((data["DAYS_BIRTH"]) * (np.minimum(((np.minimum(((data["FLAG_DOCUMENT_10"])), (((-1.0*((np.maximum(((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])), ((np.minimum(((data["AMT_GOODS_PRICE_x"])), ((data["DAYS_BIRTH"])))))))))))))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))))) 
    v["i267"] = np.tanh((((((data["AMT_CREDIT_x"]) - ((((-3.0) + (((data["cnt_REG_CITY_NOT_LIVE_CITY"]) * 2.0)))/2.0)))) > ((4.65822219848632812)))*1.)) 
    v["i268"] = np.tanh(((np.where(data["te_REG_CITY_NOT_WORK_CITY"]>0, data["FLAG_DOCUMENT_10"], (((data["te_FLAG_WORK_PHONE"]) > (((-3.0) * ((((data["cnt_FLAG_DOCUMENT_10"]) + (data["OCCUPATION_TYPE"]))/2.0)))))*1.) )) / 2.0)) 
    v["i269"] = np.tanh(np.tanh(((-1.0*((np.tanh(((((((np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["AMT_REQ_CREDIT_BUREAU_MON"], 2.0 )) < (data["cnt_DAYS_BIRTH"]))*1.)) * (((((data["cnt_DAYS_BIRTH"]) * 2.0)) / 2.0))))))))))) 
    v["i270"] = np.tanh(np.where(((((1.0) - (data["AMT_CREDIT_SUM_DEBT"]))) + (data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"]))>0, ((((((data["AMT_CREDIT_SUM_DEBT"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) - (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) * 2.0), ((data["NAME_GOODS_CATEGORY_Insurance"]) / 2.0) )) 
    v["i271"] = np.tanh((((np.minimum(((((((data["ca__Closed"]) + (data["ca__Active"]))) / 2.0))), ((((((data["ca__Active"]) / 2.0)) + (data["AMT_CREDIT_SUM"])))))) < (-1.0))*1.)) 
    v["i272"] = np.tanh((((data["AMT_CREDIT_SUM"]) < ((((((data["NAME_GOODS_CATEGORY_Insurance"]) * 2.0)) + (np.tanh((((np.where(np.tanh((2.0))>0, data["NAME_GOODS_CATEGORY_Insurance"], data["cnt_FLAG_DOCUMENT_10"] )) - (np.tanh((3.0))))))))/2.0)))*1.)) 
    v["i273"] = np.tanh(np.where(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["SK_ID_BUREAU"], data["AMT_CREDIT_SUM"] )>0, ((data["SK_ID_BUREAU"]) + ((((data["AMT_CREDIT_SUM"]) < (data["SK_ID_BUREAU"]))*1.))), data["cnt_FLAG_DOCUMENT_10"] )) 
    v["i274"] = np.tanh(np.minimum(((np.minimum(((data["CODE_REJECT_REASON_VERIF"])), (((((11.77319049835205078)) - (data["CODE_REJECT_REASON_VERIF"]))))))), ((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) * (((data["inst_NUM_INSTALMENT_NUMBER"]) * 2.0))))))) 
    v["i275"] = np.tanh(((data["OWN_CAR_AGE"]) * (np.maximum(((data["inst_AMT_PAYMENT"])), ((((np.where((-1.0*((data["inst_AMT_PAYMENT"])))>0, -3.0, (((5.90897226333618164)) * 2.0) )) * (np.tanh((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) / 2.0))))))))))) 
    v["i276"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["AMT_REQ_CREDIT_BUREAU_MON"], ((data["inst_AMT_PAYMENT"]) * ((((np.tanh((data["AMT_REQ_CREDIT_BUREAU_MON"]))) + (((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["inst_AMT_PAYMENT"]))) / 2.0)))/2.0))) )) 
    v["i277"] = np.tanh(((2.0) * (((np.where((-1.0*((data["cc_bal_SK_DPD_DEF"])))>0, data["FLAG_DOCUMENT_10"], data["inst_NUM_INSTALMENT_NUMBER"] )) * (np.maximum(((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), ((np.where(3.0>0, data["inst_NUM_INSTALMENT_NUMBER"], (1.0) ))))))))) 
    v["i278"] = np.tanh(((((((3.0)) < (data["inst_SK_ID_PREV"]))*1.)) - (((np.maximum(((np.maximum(((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), ((data["FLAG_DOCUMENT_10"]))))), ((np.tanh((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])))))) / 2.0)))) 
    v["i279"] = np.tanh(((np.minimum(((data["cnt_FLAG_DOCUMENT_10"])), ((((1.0) - ((((((((0.0) - (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))) + (data["cc_bal_cc_bal_status__Sent_proposal"]))/2.0)) * ((8.0))))))))) - (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))) 
    v["i280"] = np.tanh(np.maximum(((np.where((((3.0) > (((data["NAME_YIELD_GROUP_XNA"]) * 2.0)))*1.)>0, (((((data["NAME_YIELD_GROUP_XNA"]) - ((-1.0*((0.0)))))) > ((5.0)))*1.), data["NAME_PORTFOLIO_Cards"] ))), ((-1.0)))) 
    v["i281"] = np.tanh(((np.where((((data["NAME_CONTRACT_TYPE_Cash_loans"]) < (data["CODE_REJECT_REASON_SCOFR"]))*1.)>0, (-1.0*((((data["avg_buro_buro_bal_status_5"]) / 2.0)))), ((data["avg_buro_buro_bal_status_5"]) * 2.0) )) * (-2.0))) 
    v["i282"] = np.tanh(((data["cc_bal_cc_bal_status__Refused"]) * ((((((((((1.0) < (data["PRODUCT_COMBINATION_Card_Street"]))*1.)) * ((6.0)))) * ((6.0)))) - (np.tanh((((data["cc_bal_cc_bal_status__Refused"]) - (data["FLAG_DOCUMENT_17"]))))))))) 
    v["i283"] = np.tanh(((np.minimum(((data["DAYS_FIRST_DRAWING"])), ((((((data["AMT_REQ_CREDIT_BUREAU_MON"]) * (np.where(data["NAME_CONTRACT_STATUS_Approved"]>0, data["DAYS_FIRST_DRAWING"], ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["AMT_REQ_CREDIT_BUREAU_MON"]))) / 2.0) )))) * 2.0))))) * (data["AMT_REQ_CREDIT_BUREAU_MON"]))) 
    v["i284"] = np.tanh(((((((data["cnt_FLAG_DOCUMENT_10"]) - ((-1.0*((np.where(data["cc_bal_SK_DPD_DEF"]>0, -1.0, data["NAME_GOODS_CATEGORY_Direct_Sales"] ))))))) + (np.where(data["cc_bal_SK_DPD_DEF"]>0, -1.0, data["cnt_FLAG_DOCUMENT_10"] )))) - (data["FLAG_DOCUMENT_17"]))) 
    v["i285"] = np.tanh(((((((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))) < (((data["cc_bal_SK_DPD_DEF"]) / 2.0)))*1.)) * ((((((((data["cc_bal_SK_DPD_DEF"]) + (((1.0) / 2.0)))/2.0)) - (data["cc_bal_SK_ID_PREV"]))) - (data["cc_bal_SK_ID_PREV"]))))) 
    v["i286"] = np.tanh((((11.08083438873291016)) * (((data["cc_bal_SK_DPD_DEF"]) * (np.minimum(((np.where(data["cc_bal_MONTHS_BALANCE"]>0, (11.08083438873291016), np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"])), ((-3.0))) ))), ((np.tanh((data["cc_bal_MONTHS_BALANCE"])))))))))) 
    v["i287"] = np.tanh(((((data["cc_bal_SK_DPD"]) * (((((((data["AMT_GOODS_PRICE_x"]) * ((3.0)))) * 2.0)) * (((((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) < (0.0))*1.)) > (data["cc_bal_AMT_PAYMENT_CURRENT"]))*1.)) * 2.0)))))) / 2.0)) 
    v["i288"] = np.tanh((((((((np.minimum(((data["AMT_CREDIT_x"])), ((data["cc_bal_SK_DPD_DEF"])))) + (((((((((((data["cc_bal_SK_DPD_DEF"]) / 2.0)) + (data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]))/2.0)) + (data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]))/2.0)) - (data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"]))))/2.0)) / 2.0)) / 2.0)) 
    v["i289"] = np.tanh((((((data["cnt_FLAG_DOCUMENT_10"]) + (((((((8.0)) < ((((((data["cc_bal_SK_DPD_DEF"]) + (data["NAME_GOODS_CATEGORY_Direct_Sales"]))/2.0)) / 2.0)))*1.)) * ((((9.73687171936035156)) * 2.0)))))/2.0)) + (data["cc_bal_cc_bal_status__Refused"]))) 
    v["i290"] = np.tanh((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > (np.where(data["PRODUCT_COMBINATION_POS_industry_without_interest"]>0, data["cc_bal_AMT_PAYMENT_CURRENT"], (((((3.0) - (data["cc_bal_AMT_PAYMENT_CURRENT"]))) + ((3.74398207664489746)))/2.0) )))*1.)) 
    v["i291"] = np.tanh((((np.minimum((((((((10.36883640289306641)) - (((data["PRODUCT_COMBINATION_POS_industry_without_interest"]) * 2.0)))) * 2.0))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) + (((np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["avg_buro_buro_bal_status_5"], (-1.0*((data["avg_buro_buro_bal_status_5"]))) )) / 2.0)))/2.0)) 
    v["i292"] = np.tanh(((np.where(((data["avg_buro_buro_bal_status_3"]) * (((data["cnt_FLAG_DOCUMENT_15"]) - (data["NAME_GOODS_CATEGORY_Other"]))))>0, data["avg_buro_buro_bal_status_5"], np.minimum(((((data["cnt_FLAG_DOCUMENT_15"]) / 2.0))), ((((data["cnt_FLAG_DOCUMENT_15"]) - (data["NAME_GOODS_CATEGORY_Other"]))))) )) * 2.0)) 
    v["i293"] = np.tanh(((data["NAME_GOODS_CATEGORY_Other"]) * (((np.tanh((np.where(-3.0>0, (((14.68852424621582031)) / 2.0), ((data["NAME_GOODS_CATEGORY_Other"]) - ((((14.23720550537109375)) / 2.0))) )))) / 2.0)))) 
    v["i294"] = np.tanh(np.where((((6.69561433792114258)) - (data["NAME_GOODS_CATEGORY_Other"]))>0, ((((((7.0)) < (data["NAME_GOODS_CATEGORY_Other"]))*1.)) - ((-1.0*(((-1.0*((np.maximum(((data["NAME_SELLER_INDUSTRY_MLM_partners"])), ((data["NAME_GOODS_CATEGORY_Weapon"]))))))))))), data["NAME_GOODS_CATEGORY_Other"] )) 
    v["i295"] = np.tanh(np.minimum(((((((np.where(data["NAME_GOODS_CATEGORY_Auto_Accessories"]>0, (-1.0*((3.0))), np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Furniture"])), ((data["NAME_GOODS_CATEGORY_Auto_Accessories"]))) )) / 2.0)) * (data["DAYS_LAST_DUE_1ST_VERSION"])))), ((data["cnt_FLAG_DOCUMENT_4"])))) 
    v["i296"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["cc_bal_CNT_DRAWINGS_POS_CURRENT"], (((((np.where(np.where((7.0)>0, data["DAYS_FIRST_DUE"], data["DAYS_FIRST_DUE"] )>0, data["NAME_GOODS_CATEGORY_Insurance"], -1.0 )) > (((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) * 2.0)))*1.)) * 2.0) )) 
    v["i297"] = np.tanh((((7.0)) * (np.tanh((((np.where(data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]>0, ((data["AMT_CREDIT_MAX_OVERDUE"]) / 2.0), (-1.0*((((np.maximum(((data["AMT_CREDIT_MAX_OVERDUE"])), ((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"])))) * 2.0)))) )) * 2.0)))))) 
    v["i298"] = np.tanh(((data["AMT_GOODS_PRICE_x"]) * (np.minimum(((data["NAME_GOODS_CATEGORY_Fitness"])), ((np.minimum(((data["FLAG_DOCUMENT_10"])), (((((((5.34786462783813477)) - (data["NAME_GOODS_CATEGORY_Fitness"]))) / 2.0)))))))))) 
    v["i299"] = np.tanh(np.where(np.maximum(((((data["DAYS_ENDDATE_FACT"]) * (((((data["DAYS_ENDDATE_FACT"]) * ((4.0)))) / 2.0))))), ((((data["AMT_CREDIT_x"]) + (-3.0)))))>0, data["FLAG_DOCUMENT_10"], ((data["cc_bal_CNT_DRAWINGS_CURRENT"]) * 2.0) )) 
    v["i300"] = np.tanh((((data["AMT_CREDIT_x"]) > (((3.0) - (((data["DAYS_FIRST_DUE"]) - (np.maximum((((-1.0*((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))))), ((np.minimum(((-3.0)), (((-1.0*((data["cnt_FLAG_DOCUMENT_10"])))))))))))))))*1.)) 
    v["i301"] = np.tanh(((((((3.0) + ((((1.0) < (((data["NAME_EDUCATION_TYPE"]) / 2.0)))*1.)))/2.0)) < ((-1.0*((np.minimum(((data["cc_bal_AMT_DRAWINGS_CURRENT"])), ((((data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]) * 2.0)))))))))*1.)) 
    v["i302"] = np.tanh(np.minimum(((((((np.tanh((np.minimum(((data["NAME_PAYMENT_TYPE_Cash_through_the_bank"])), ((0.0)))))) / 2.0)) / 2.0))), ((np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]>0, data["NAME_PAYMENT_TYPE_XNA"], 0.0 ))))) 
    v["i303"] = np.tanh(np.tanh((np.minimum(((((data["PRODUCT_COMBINATION_Cash"]) * (((data["SK_ID_PREV_x"]) - (data["AMT_GOODS_PRICE_x"])))))), ((((((((1.0) > ((-1.0*((data["PRODUCT_COMBINATION_Cash"])))))*1.)) + (data["SK_ID_PREV_x"]))/2.0))))))) 
    v["i304"] = np.tanh((-1.0*(((((((((3.0) - (data["AMT_GOODS_PRICE_x"]))) / 2.0)) > (((3.0) - (np.maximum(((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])), ((data["AMT_GOODS_PRICE_x"])))))))*1.))))) 
    v["i305"] = np.tanh(np.where(data["cc_bal_AMT_PAYMENT_CURRENT"]>0, data["DAYS_ENDDATE_FACT"], ((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) * ((((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) < (data["DAYS_ENDDATE_FACT"]))*1.)) / 2.0)))) * ((-1.0*((data["cc_bal_AMT_PAYMENT_CURRENT"]))))) )) 
    v["i306"] = np.tanh(((((2.0) * 2.0)) * (np.where(((data["NAME_TYPE_SUITE_Unaccompanied"]) / 2.0)>0, data["NAME_CASH_LOAN_PURPOSE_Furniture"], ((data["cnt_FLAG_DOCUMENT_10"]) * 2.0) )))) 
    v["i307"] = np.tanh(((np.tanh((((((data["NAME_CONTRACT_STATUS_Approved"]) * ((((data["DAYS_LAST_DUE_1ST_VERSION"]) > ((((-1.0*((1.0)))) / 2.0)))*1.)))) / 2.0)))) - (((data["NAME_CONTRACT_STATUS_Approved"]) * (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))) 
    v["i308"] = np.tanh(np.where(data["SK_ID_PREV_x"]>0, ((((((data["cc_bal_AMT_RECIVABLE"]) < (data["NAME_GOODS_CATEGORY_Fitness"]))*1.)) + (data["NAME_GOODS_CATEGORY_Fitness"]))/2.0), data["cc_bal_AMT_RECIVABLE"] )) 
    v["i309"] = np.tanh(((((8.43800640106201172)) < (np.where((5.17690801620483398)>0, (((((11.66382503509521484)) / 2.0)) - ((-1.0*(((-1.0*((((data["DAYS_LAST_DUE_1ST_VERSION"]) - (data["CODE_REJECT_REASON_LIMIT"])))))))))), data["cc_bal_SK_DPD"] )))*1.)) 
    v["i310"] = np.tanh(((data["cc_bal_SK_DPD"]) * ((-1.0*((np.minimum(((data["DAYS_LAST_DUE_1ST_VERSION"])), ((np.tanh((((np.minimum(((2.0)), ((data["cc_bal_SK_DPD"])))) - (data["DAYS_LAST_DUE_1ST_VERSION"]))))))))))))) 
    v["i311"] = np.tanh(((((((np.tanh((1.0))) + ((((data["FLAG_DOCUMENT_16"]) + (2.0))/2.0)))) * ((5.0)))) * (((data["NAME_GOODS_CATEGORY_Insurance"]) * (((data["FLAG_DOCUMENT_16"]) - (data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]))))))) 
    v["i312"] = np.tanh(((((data["NAME_SELLER_INDUSTRY_Tourism"]) * (np.where((((data["NAME_SELLER_INDUSTRY_Tourism"]) < (data["NAME_TYPE_SUITE_Group_of_people"]))*1.)>0, ((data["NAME_TYPE_SUITE_Group_of_people"]) * 2.0), data["PRODUCT_COMBINATION_POS_mobile_without_interest"] )))) * 2.0)) 
    v["i313"] = np.tanh((((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, np.minimum(((data["CNT_INSTALMENT"])), ((((data["NAME_GOODS_CATEGORY_Insurance"]) * (data["FLAG_DOCUMENT_4"]))))), data["NAME_GOODS_CATEGORY_Education"] )) < (np.minimum(((-1.0)), ((data["NAME_GOODS_CATEGORY_Education"])))))*1.)) * 2.0)) 
    v["i314"] = np.tanh(((((((np.where(((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) / 2.0)>0, data["NAME_CASH_LOAN_PURPOSE_Furniture"], data["CNT_INSTALMENT"] )) * (data["NAME_GOODS_CATEGORY_Education"]))) - ((((data["FLAG_DOCUMENT_4"]) + (data["FLAG_DOCUMENT_10"]))/2.0)))) * 2.0)) 
    v["i315"] = np.tanh(np.where(((data["PRODUCT_COMBINATION_Cash"]) - (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))>0, np.minimum(((np.tanh((data["cnt_FLAG_DOCUMENT_10"])))), ((((data["CNT_INSTALMENT_FUTURE"]) / 2.0)))), (((-1.0*(((((data["NAME_GOODS_CATEGORY_Fitness"]) > (1.0))*1.))))) / 2.0) )) 
    v["i316"] = np.tanh(((((np.tanh((np.where(data["NAME_YIELD_GROUP_XNA"]>0, data["PRODUCT_COMBINATION_Cash"], data["NAME_CONTRACT_TYPE_Cash_loans"] )))) / 2.0)) / 2.0)) 
    v["i317"] = np.tanh(np.minimum((((-1.0*(((((np.where(-1.0>0, 2.0, ((data["NAME_SELLER_INDUSTRY_XNA"]) + (data["NAME_PORTFOLIO_Cash"])) )) > ((3.0)))*1.)))))), ((data["NAME_GOODS_CATEGORY_Insurance"])))) 
    v["i318"] = np.tanh(np.where(data["CHANNEL_TYPE_AP___Cash_loan_"]>0, (((((data["CHANNEL_TYPE_AP___Cash_loan_"]) / 2.0)) > (((np.tanh((np.minimum(((0.0)), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"])))))) + (2.0))))*1.), (-1.0*((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * 2.0)))) )) 
    v["i319"] = np.tanh(np.maximum((((((np.where(np.where(data["DAYS_LAST_DUE_1ST_VERSION"]>0, data["CHANNEL_TYPE_Stone"], data["cnt_FLAG_DOCUMENT_10"] )>0, 3.0, data["CNT_INSTALMENT"] )) < (((data["NAME_GOODS_CATEGORY_Insurance"]) / 2.0)))*1.))), (((-1.0*((0.0))))))) 
    v["i320"] = np.tanh((-1.0*(((((np.where(((data["NAME_PORTFOLIO_Cards"]) + (data["DAYS_FIRST_DRAWING"]))>0, data["NAME_YIELD_GROUP_XNA"], (-1.0*((data["NAME_GOODS_CATEGORY_Insurance"]))) )) > ((2.0)))*1.))))) 
    v["i321"] = np.tanh(np.tanh((np.where(data["DAYS_LAST_DUE_1ST_VERSION"]>0, ((((data["EXT_SOURCE_3"]) * 2.0)) - (1.0)), data["NAME_GOODS_CATEGORY_Insurance"] )))) 
    v["i322"] = np.tanh(((((np.tanh((((((((data["ca__Active"]) * 2.0)) * ((4.0)))) * 2.0)))) / 2.0)) / 2.0)) 
    v["i323"] = np.tanh((((data["DAYS_CREDIT_ENDDATE"]) < ((-1.0*(((((((data["NAME_GOODS_CATEGORY_Fitness"]) > (data["DAYS_CREDIT_ENDDATE"]))*1.)) / 2.0))))))*1.)) 
    v["i324"] = np.tanh((((((((np.where((((1.0) + (data["CNT_INSTALMENT_FUTURE"]))/2.0)>0, data["DAYS_CREDIT"], -2.0 )) + (data["DAYS_CREDIT"]))) * (data["SK_ID_BUREAU"]))) + (data["DAYS_CREDIT"]))/2.0)) 
    v["i325"] = np.tanh(((data["SK_ID_BUREAU"]) * ((((((((data["CNT_INSTALMENT_FUTURE"]) * 2.0)) < (data["SK_ID_BUREAU"]))*1.)) * (data["AMT_CREDIT_x"]))))) 
    v["i326"] = np.tanh((((((np.where(data["SK_ID_BUREAU"]>0, data["CNT_INSTALMENT_FUTURE"], data["FLAG_DOCUMENT_2"] )) + (np.where(data["SK_ID_BUREAU"]>0, data["cnt_DAYS_ID_PUBLISH"], np.tanh(((((1.0) > ((11.27696132659912109)))*1.))) )))/2.0)) / 2.0)) 
    v["i327"] = np.tanh(np.minimum(((0.0)), ((((data["DAYS_CREDIT"]) * (np.maximum(((data["CNT_INSTALMENT"])), ((np.maximum(((data["inst_NUM_INSTALMENT_VERSION"])), ((np.tanh(((((-2.0) > (0.0))*1.))))))))))))))) 
    v["i328"] = np.tanh(np.tanh(((((np.tanh((np.tanh(((-1.0*((np.maximum(((-3.0)), ((data["ca__Closed"]))))))))))) + (((data["ca__Closed"]) * ((-1.0*((data["SK_ID_BUREAU"])))))))/2.0)))) 
    v["i329"] = np.tanh(np.maximum(((((((((np.where(data["DAYS_ENDDATE_FACT"]>0, -1.0, data["SK_ID_BUREAU"] )) + (1.0))) * (data["AMT_CREDIT_y"]))) * (data["CNT_INSTALMENT_FUTURE"])))), ((0.0)))) 
    v["i330"] = np.tanh(((((((data["inst_AMT_PAYMENT"]) * (((((data["SK_ID_BUREAU"]) / 2.0)) * (data["inst_AMT_PAYMENT"]))))) + ((((0.0) < (data["PRODUCT_COMBINATION_POS_household_without_interest"]))*1.)))) / 2.0)) 
    v["i331"] = np.tanh(np.minimum((((((data["AMT_CREDIT_SUM_DEBT"]) + (np.maximum(((data["EXT_SOURCE_3"])), ((data["AMT_CREDIT_SUM_DEBT"])))))/2.0))), (((((0.0)) - (np.where(data["AMT_CREDIT_SUM_DEBT"]>0, data["EXT_SOURCE_3"], data["NAME_GOODS_CATEGORY_Insurance"] ))))))) 
    v["i332"] = np.tanh((-1.0*((((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) + ((((np.tanh((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((data["EXT_SOURCE_3"]) / 2.0)))))) < (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))*1.)))) * (data["EXT_SOURCE_3"])))))) 
    v["i333"] = np.tanh(((((np.minimum(((((data["DAYS_LAST_DUE"]) * (data["CNT_INSTALMENT"])))), (((((10.0)) * (1.0)))))) / 2.0)) * (data["inst_AMT_PAYMENT"]))) 
    v["i334"] = np.tanh(((data["EXT_SOURCE_3"]) * ((((((data["AMT_GOODS_PRICE_x"]) < (((np.tanh((((data["AMT_GOODS_PRICE_x"]) * 2.0)))) * 2.0)))*1.)) / 2.0)))) 
    v["i335"] = np.tanh(((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((((data["DAYS_CREDIT_UPDATE"]) < (-3.0))*1.))))) * (data["DAYS_CREDIT"]))) 
    v["i336"] = np.tanh(np.minimum(((((np.minimum((((-1.0*((data["CHANNEL_TYPE_Stone"]))))), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))) * (((data["inst_AMT_PAYMENT"]) / 2.0))))), ((((((-1.0*((data["NAME_YIELD_GROUP_low_action"])))) + (2.0))/2.0))))) 
    v["i337"] = np.tanh(((data["NAME_CASH_LOAN_PURPOSE_Repairs"]) * (((((data["NAME_CASH_LOAN_PURPOSE_Repairs"]) * ((((((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((((data["EXT_SOURCE_3"]) + (1.0))/2.0))))) + (data["inst_DAYS_INSTALMENT"]))/2.0)) / 2.0)))) / 2.0)))) 
    v["i338"] = np.tanh(np.where(np.maximum(((data["te_DAYS_ID_PUBLISH"])), ((np.tanh((data["te_DAYS_ID_PUBLISH"])))))>0, (((((data["te_DAYS_ID_PUBLISH"]) > (3.0))*1.)) * (data["te_DAYS_ID_PUBLISH"])), data["AMT_CREDIT_x"] )) 
    v["i339"] = np.tanh(((data["NAME_GOODS_CATEGORY_Medicine"]) + ((((data["NAME_GOODS_CATEGORY_Medicine"]) > ((((((data["EXT_SOURCE_3"]) + (2.0))/2.0)) + ((-1.0*((np.tanh((data["NAME_GOODS_CATEGORY_Medicine"])))))))))*1.)))) 
    v["i340"] = np.tanh(np.where((-1.0*((((((((((0.41248092055320740)) / 2.0)) + (data["AMT_GOODS_PRICE_x"]))/2.0)) * (data["AMT_GOODS_PRICE_x"])))))>0, -2.0, ((((data["cnt_FLAG_DOCUMENT_10"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) * 2.0) )) 
    v["i341"] = np.tanh(((((data["inst_AMT_PAYMENT"]) * (((np.where(0.0>0, (-1.0*((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))), np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((((-2.0) + (((data["inst_AMT_PAYMENT"]) / 2.0)))))) )) * 2.0)))) * 2.0)) 
    v["i342"] = np.tanh(((np.where(data["CNT_INSTALMENT_FUTURE"]>0, (((-1.0*((2.0)))) + (((data["CODE_REJECT_REASON_VERIF"]) - (((data["CODE_REJECT_REASON_VERIF"]) * 2.0))))), np.tanh((data["inst_AMT_PAYMENT"])) )) * (data["CODE_REJECT_REASON_VERIF"]))) 
    v["i343"] = np.tanh(np.where((((data["NAME_CASH_LOAN_PURPOSE_Repairs"]) < (((data["AMT_DOWN_PAYMENT"]) * 2.0)))*1.)>0, data["NAME_CASH_LOAN_PURPOSE_Repairs"], np.tanh(((-1.0*((((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) + ((5.25137329101562500)))/2.0)) < (data["NAME_CASH_LOAN_PURPOSE_Repairs"]))*1.)))))) )) 
    v["i344"] = np.tanh(((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) + (-1.0))/2.0)) > ((((np.maximum(((-1.0)), ((((data["inst_AMT_PAYMENT"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"])))))) + (data["AMT_DOWN_PAYMENT"]))/2.0)))*1.)) 
    v["i345"] = np.tanh(((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (((((np.tanh((np.where((((data["AMT_GOODS_PRICE_x"]) > (data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]))*1.)>0, data["inst_AMT_PAYMENT"], -2.0 )))) * 2.0)) * 2.0)))) 
    v["i346"] = np.tanh(((np.maximum(((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) - (-1.0)))), ((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])))) * (np.tanh(((-1.0*((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) - (data["cc_bal_AMT_DRAWINGS_CURRENT"]))))))))))))) 
    v["i347"] = np.tanh(((((data["inst_AMT_INSTALMENT"]) * 2.0)) * ((((((0.0) / 2.0)) > ((((((data["AMT_CREDIT_x"]) + (1.0))) + (((data["inst_AMT_INSTALMENT"]) / 2.0)))/2.0)))*1.)))) 
    v["i348"] = np.tanh(((np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, (((data["inst_AMT_PAYMENT"]) < (data["AMT_GOODS_PRICE_x"]))*1.), data["NAME_GOODS_CATEGORY_Direct_Sales"] )) * (((data["inst_AMT_PAYMENT"]) + (((-2.0) * 2.0)))))) 
    v["i349"] = np.tanh(((data["AMT_GOODS_PRICE_x"]) * (np.minimum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((((data["NAME_GOODS_CATEGORY_Insurance"]) - ((((data["cu__currency_3"]) > ((-1.0*((np.tanh((data["NAME_GOODS_CATEGORY_Direct_Sales"])))))))*1.))))))))) 
    v["i350"] = np.tanh(np.maximum(((((((data["AMT_DOWN_PAYMENT"]) * (data["cu__currency_3"]))) * (-3.0)))), (((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) > (np.maximum((((14.77923965454101562))), ((data["AMT_DOWN_PAYMENT"])))))*1.))))) 
    v["i351"] = np.tanh((((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) * 2.0)) < (((((((-1.0*((2.0)))) + (((data["AMT_DOWN_PAYMENT"]) / 2.0)))/2.0)) / 2.0)))*1.)) 
    v["i352"] = np.tanh((((((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]) - (((data["CHANNEL_TYPE_Car_dealer"]) / 2.0)))) + ((((((data["CHANNEL_TYPE_Car_dealer"]) - (data["NAME_GOODS_CATEGORY_Insurance"]))) > (data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]))*1.)))/2.0)) 
    v["i353"] = np.tanh((-1.0*((np.maximum(((((np.where(data["NAME_GOODS_CATEGORY_Fitness"]>0, data["NAME_GOODS_CATEGORY_Fitness"], data["NAME_GOODS_CATEGORY_Weapon"] )) + (np.tanh((data["FLAG_DOCUMENT_10"])))))), (((((((data["NAME_GOODS_CATEGORY_Fitness"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) < (-1.0))*1.)))))))) 
    v["i354"] = np.tanh(((data["cnt_FLAG_DOCUMENT_9"]) * ((((data["cnt_FLAG_DOCUMENT_4"]) + (np.where(np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((data["cnt_FLAG_DOCUMENT_9"])))>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], -3.0 )))/2.0)))) 
    v["i355"] = np.tanh(np.minimum(((data["FLAG_DOCUMENT_17"])), ((((np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((((data["YEARS_BUILD_AVG"]) + (data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"]))/2.0))))) + ((-1.0*((np.where((-1.0*((data["te_FLAG_DOCUMENT_8"])))>0, data["YEARS_BUILD_AVG"], data["FLAG_DOCUMENT_17"] )))))))))) 
    v["i356"] = np.tanh(np.where((10.90770244598388672)>0, ((((((2.0)) < (data["HOUSETYPE_MODE"]))*1.)) * (((data["nans"]) * (((-3.0) * 2.0))))), (-1.0*((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) )) 
    v["i357"] = np.tanh(((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((((data["cnt_WALLSMATERIAL_MODE"]) - (np.minimum(((3.0)), (((-1.0*((((data["te_FONDKAPREMONT_MODE"]) * 2.0)))))))))) - ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) < (1.0))*1.)))))) 
    v["i358"] = np.tanh(np.tanh(((((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["te_FLAG_DOCUMENT_8"])))) < (((data["inst_AMT_PAYMENT"]) - ((((((((3.0) + (data["inst_AMT_PAYMENT"]))) + (data["cnt_HOUSETYPE_MODE"]))/2.0)) * 2.0)))))*1.)))) 
    v["i359"] = np.tanh(np.minimum(((((((data["cnt_FLAG_DOCUMENT_8"]) * (((1.0) - (np.maximum(((data["AMT_GOODS_PRICE_x"])), ((np.maximum(((data["FLAG_DOCUMENT_8"])), ((0.0))))))))))) / 2.0))), ((((data["cnt_FLAG_DOCUMENT_8"]) * (data["cnt_HOUSETYPE_MODE"])))))) 
    v["i360"] = np.tanh(((((np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["te_REG_REGION_NOT_LIVE_REGION"], np.maximum(((data["CODE_REJECT_REASON_SCOFR"])), ((data["NAME_GOODS_CATEGORY_Insurance"]))) )) * (data["te_REG_REGION_NOT_LIVE_REGION"]))) * 2.0)) 
    v["i361"] = np.tanh(np.where(data["AMT_CREDIT_MAX_OVERDUE"]>0, data["AMT_CREDIT_MAX_OVERDUE"], (((((data["AMT_CREDIT_MAX_OVERDUE"]) * 2.0)) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.) )) 
    v["i362"] = np.tanh(((((-1.0*((((data["te_REG_REGION_NOT_LIVE_REGION"]) * (((((((3.0) * (data["te_FLAG_EMAIL"]))) / 2.0)) / 2.0))))))) + (np.where(data["te_FLAG_EMAIL"]>0, 0.0, data["AMT_CREDIT_x"] )))/2.0)) 
    v["i363"] = np.tanh((((((((np.minimum(((-2.0)), (((((data["te_FLAG_EMAIL"]) < (np.tanh(((7.92169857025146484)))))*1.))))) - (3.0))) > (data["te_FLAG_EMAIL"]))*1.)) - (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0)))) 
    v["i364"] = np.tanh(np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((((data["CNT_INSTALMENT"]) * ((((((data["AMT_GOODS_PRICE_x"]) * (data["cnt_DAYS_BIRTH"]))) + (data["cnt_DAYS_BIRTH"]))/2.0))))))) 
    v["i365"] = np.tanh(((np.where(data["NAME_GOODS_CATEGORY_Education"]>0, ((data["AMT_CREDIT_x"]) - (((((data["NAME_GOODS_CATEGORY_Education"]) + (data["cnt_FLAG_DOCUMENT_18"]))) * 2.0))), ((data["cnt_FLAG_DOCUMENT_18"]) - (((data["NAME_GOODS_CATEGORY_Education"]) * 2.0))) )) / 2.0)) 
    v["i366"] = np.tanh(((data["cnt_FLAG_DOCUMENT_14"]) * (np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["FLAG_DOCUMENT_18"], ((data["FLAG_DOCUMENT_18"]) + (((data["FLAG_DOCUMENT_14"]) - (((1.0) - (((data["cnt_FLAG_DOCUMENT_18"]) * ((7.0))))))))) )))) 
    v["i367"] = np.tanh(np.where((((data["CNT_INSTALMENT_FUTURE"]) > (np.minimum(((1.0)), ((1.0)))))*1.)>0, data["nans"], (((data["cnt_FLAG_DOCUMENT_14"]) + (np.tanh(((-1.0*((data["nans"])))))))/2.0) )) 
    v["i368"] = np.tanh(((data["DAYS_CREDIT"]) * (np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((((((data["CNT_INSTALMENT_FUTURE"]) + ((((-1.0*((((data["te_FLAG_DOCUMENT_20"]) * (data["WALLSMATERIAL_MODE"])))))) * (((data["CNT_INSTALMENT_FUTURE"]) * 2.0)))))/2.0)) / 2.0))))))) 
    v["i369"] = np.tanh((((((3.0) / 2.0)) < (((data["avg_buro_buro_bal_status_C"]) - ((((np.where(data["AMT_CREDIT_y"]>0, data["te_FLAG_DOCUMENT_20"], data["avg_buro_MONTHS_BALANCE"] )) + (data["WALLSMATERIAL_MODE"]))/2.0)))))*1.)) 
    v["i370"] = np.tanh(((((data["WALLSMATERIAL_MODE"]) * (((np.where(2.0>0, ((data["AMT_CREDIT_y"]) * 2.0), -3.0 )) * (data["NAME_GOODS_CATEGORY_Auto_Accessories"]))))) - ((((data["te_FLAG_DOCUMENT_20"]) > (np.tanh((2.0))))*1.)))) 
    v["i371"] = np.tanh((-1.0*((((((2.71130990982055664)) < ((((((data["NAME_GOODS_CATEGORY_Auto_Accessories"]) - ((((-1.0*((1.0)))) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))) + (((1.0) + (data["AMT_CREDIT_y"]))))/2.0)))*1.))))) 
    v["i372"] = np.tanh(((((((-1.0) + (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"]) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"]))))/2.0)) < (((((data["AMT_CREDIT_y"]) * 2.0)) - ((((3.0)) * 2.0)))))*1.)) 
    v["i373"] = np.tanh(np.minimum(((((data["AMT_GOODS_PRICE_x"]) * (((data["inst_AMT_PAYMENT"]) - (data["NAME_CASH_LOAN_PURPOSE_Business_development"])))))), ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (np.maximum(((data["inst_AMT_PAYMENT"])), ((((np.tanh((data["CNT_INSTALMENT_FUTURE"]))) / 2.0)))))))))) 
    v["i374"] = np.tanh(((((((data["CNT_INSTALMENT_FUTURE"]) * ((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"])))))) * 2.0)) + (((-2.0) * (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * ((((2.0) > (-3.0))*1.)))))))) 
    v["i375"] = np.tanh(((((np.maximum((((-1.0*((data["inst_AMT_PAYMENT"]))))), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"])))) * (data["inst_AMT_PAYMENT"]))) * (data["CNT_INSTALMENT"]))) 
    v["i376"] = np.tanh(((data["inst_AMT_INSTALMENT"]) * ((((((np.where(-2.0>0, ((-1.0) * (data["inst_AMT_INSTALMENT"])), data["CNT_INSTALMENT_FUTURE"] )) < ((((data["NAME_YIELD_GROUP_high"]) + (data["inst_AMT_INSTALMENT"]))/2.0)))*1.)) / 2.0)))) 
    v["i377"] = np.tanh(((((((data["inst_AMT_INSTALMENT"]) - (data["inst_AMT_PAYMENT"]))) * 2.0)) * ((4.81916522979736328)))) 
    v["i378"] = np.tanh((-1.0*(((((((((((((np.where(data["AMT_CREDIT_y"]>0, (0.0), data["inst_AMT_PAYMENT"] )) > (data["inst_AMT_INSTALMENT"]))*1.)) / 2.0)) > ((((0.0) > ((11.85620403289794922)))*1.)))*1.)) + (data["inst_AMT_PAYMENT"]))/2.0))))) 
    v["i379"] = np.tanh((-1.0*((((((np.where(((data["NAME_PORTFOLIO_Cash"]) + ((-1.0*((np.minimum(((np.tanh((data["AMT_ANNUITY"])))), ((data["inst_AMT_PAYMENT"]))))))))>0, data["AMT_ANNUITY"], 0.0 )) / 2.0)) / 2.0))))) 
    v["i380"] = np.tanh(np.where(data["cc_bal_SK_ID_PREV"]>0, np.tanh((1.0)), ((data["inst_NUM_INSTALMENT_NUMBER"]) * ((((((1.0) < (data["inst_NUM_INSTALMENT_NUMBER"]))*1.)) + (data["inst_NUM_INSTALMENT_NUMBER"])))) )) 
    v["i381"] = np.tanh(((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((data["avg_buro_buro_bal_status_1"]) * (((data["inst_NUM_INSTALMENT_VERSION"]) - (data["PRODUCT_COMBINATION_Cash_X_Sell__high"])))))))) + ((((((data["inst_NUM_INSTALMENT_VERSION"]) * 2.0)) > (1.0))*1.)))) 
    v["i382"] = np.tanh(np.minimum(((((data["Completed"]) * (((data["PRODUCT_COMBINATION_Cash_Street__high"]) * (1.0)))))), ((np.tanh((((((((((data["NFLAG_INSURED_ON_APPROVAL"]) * (data["CNT_INSTALMENT_FUTURE"]))) * 2.0)) / 2.0)) / 2.0))))))) 
    v["i383"] = np.tanh(np.maximum(((np.where(data["PRODUCT_COMBINATION_Cash_X_Sell__high"]>0, data["NAME_YIELD_GROUP_low_action"], (((np.maximum(((data["NAME_YIELD_GROUP_low_action"])), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))) > (1.0))*1.) ))), (((((data["SK_ID_PREV_y"]) < (((2.0) - ((3.0)))))*1.))))) 
    v["i384"] = np.tanh(((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["MONTHS_BALANCE"])))) * (((data["CNT_INSTALMENT_FUTURE"]) - (np.minimum(((data["inst_AMT_PAYMENT"])), (((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)) + ((14.19008731842041016))))))))))) 
    v["i385"] = np.tanh(((data["MONTHS_BALANCE"]) * (np.tanh((np.minimum((((-1.0*((data["NAME_YIELD_GROUP_low_action"]))))), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))))))) 
    v["i386"] = np.tanh((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + ((-1.0*((((((np.tanh((data["SK_ID_PREV_y"]))) / 2.0)) * 2.0))))))/2.0)) / 2.0)) 
    v["i387"] = np.tanh(np.minimum(((np.tanh((((data["NAME_SELLER_INDUSTRY_Connectivity"]) * (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (data["NAME_SELLER_INDUSTRY_Connectivity"])))))))), ((((data["NAME_SELLER_INDUSTRY_Connectivity"]) * (((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) * 2.0))))))) 
    v["i388"] = np.tanh(((data["NAME_SELLER_INDUSTRY_Connectivity"]) * ((-1.0*((((((((data["NAME_SELLER_INDUSTRY_Connectivity"]) > (data["DAYS_FIRST_DRAWING"]))*1.)) < (np.maximum(((0.0)), ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - (((data["cc_bal_AMT_PAYMENT_CURRENT"]) * 2.0))))))))*1.))))))) 
    v["i389"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, data["DAYS_FIRST_DRAWING"], ((((data["cc_bal_AMT_PAYMENT_CURRENT"]) * (data["CNT_INSTALMENT"]))) / 2.0) )) 
    v["i390"] = np.tanh(np.where(((data["DAYS_FIRST_DRAWING"]) * ((((1.90255212783813477)) * (-2.0))))>0, (((((data["cc_bal_AMT_PAYMENT_CURRENT"]) / 2.0)) > (((3.0) + (data["DAYS_FIRST_DRAWING"]))))*1.), np.tanh((data["cc_bal_AMT_PAYMENT_CURRENT"])) )) 
    v["i391"] = np.tanh(((((data["SK_ID_BUREAU"]) * 2.0)) * ((((np.tanh((3.0))) < (((data["SK_ID_BUREAU"]) + (((((((data["CNT_INSTALMENT"]) + (-2.0))) * 2.0)) * 2.0)))))*1.)))) 
    v["i392"] = np.tanh((((data["NAME_SELLER_INDUSTRY_Connectivity"]) > (((((2.0)) + (((((((3.0) / 2.0)) + (data["MONTHS_BALANCE"]))) + (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["MONTHS_BALANCE"], np.tanh((2.0)) )))))/2.0)))*1.)) 
    v["i393"] = np.tanh(np.minimum((((((((((((((data["SK_DPD"]) + (data["WALLSMATERIAL_MODE"]))/2.0)) * (data["SK_ID_PREV_x"]))) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) * 2.0)) * 2.0))), ((np.where(data["WALLSMATERIAL_MODE"]>0, data["NAME_CLIENT_TYPE_Repeater"], (2.05001640319824219) ))))) 
    v["i394"] = np.tanh((-1.0*((((((((data["CNT_INSTALMENT"]) < (2.0))*1.)) < (data["te_WALLSMATERIAL_MODE"]))*1.))))) 
    v["i395"] = np.tanh(((((((((data["CNT_INSTALMENT_FUTURE"]) / 2.0)) + (((np.tanh((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 3.0, data["cnt_WALLSMATERIAL_MODE"] )))) + (data["AMT_CREDIT_x"]))))/2.0)) > (2.0))*1.)) 
    v["i396"] = np.tanh((((((((2.0) - ((((data["WALLSMATERIAL_MODE"]) > ((11.67886257171630859)))*1.)))) - ((((data["cnt_WALLSMATERIAL_MODE"]) > ((-1.0*((data["DAYS_LAST_DUE_1ST_VERSION"])))))*1.)))) < (data["WALLSMATERIAL_MODE"]))*1.)) 
    v["i397"] = np.tanh(((np.where((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) > (data["DAYS_LAST_DUE_1ST_VERSION"]))*1.)>0, np.where(-1.0>0, np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])), ((0.0))), data["NAME_PORTFOLIO_Cards"] ), data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"] )) * (data["AMT_CREDIT_x"]))) 
    v["i398"] = np.tanh(((((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) * (((((((data["DAYS_LAST_DUE_1ST_VERSION"]) > ((((data["DAYS_LAST_DUE_1ST_VERSION"]) > (((((data["inst_NUM_INSTALMENT_VERSION"]) * 2.0)) / 2.0)))*1.)))*1.)) + (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))/2.0)))) * 2.0)) 
    v["i399"] = np.tanh(((-2.0) * (((np.tanh((data["avg_buro_buro_count"]))) * (np.where(np.where((5.0)>0, np.maximum(((data["CNT_INSTALMENT_FUTURE"])), ((data["avg_buro_buro_count"]))), data["CNT_INSTALMENT_FUTURE"] )>0, data["NAME_CASH_LOAN_PURPOSE_Business_development"], data["AMT_GOODS_PRICE_x"] )))))) 
    v["i400"] = np.tanh(np.tanh((((data["FLAG_OWN_CAR"]) * (np.minimum(((data["inst_AMT_PAYMENT"])), ((((((-1.0*((data["avg_buro_buro_count"])))) > (((data["AMT_GOODS_PRICE_x"]) * 2.0)))*1.))))))))) 
    v["i401"] = np.tanh(((3.0) * ((((-1.0*((((data["FLAG_OWN_CAR"]) + (data["te_FLAG_OWN_CAR"])))))) + ((((data["avg_buro_buro_count"]) < (((((3.0) / 2.0)) + ((-1.0*((3.0)))))))*1.)))))) 
    v["i402"] = np.tanh(((((np.tanh((((np.minimum(((((data["te_FLAG_DOCUMENT_2"]) / 2.0))), ((np.minimum(((data["te_FLAG_DOCUMENT_12"])), ((data["te_DAYS_BIRTH"]))))))) - (((np.tanh((((1.0) / 2.0)))) / 2.0)))))) / 2.0)) / 2.0)) 
    v["i403"] = np.tanh(((((((1.0) * (((np.where(data["AMT_GOODS_PRICE_x"]>0, -1.0, data["te_CODE_GENDER"] )) / 2.0)))) / 2.0)) / 2.0)) 
    v["i404"] = np.tanh((((np.where((((data["AMT_CREDIT_y"]) + (((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) / 2.0)))/2.0)>0, ((data["AMT_CREDIT_y"]) + (1.0)), np.minimum(((((2.0) / 2.0))), ((data["avg_buro_buro_count"]))) )) < (-1.0))*1.)) 
    v["i405"] = np.tanh(np.minimum(((((np.maximum(((data["CNT_INSTALMENT_FUTURE"])), (((-1.0*((data["avg_buro_MONTHS_BALANCE"]))))))) / 2.0))), (((((data["FLAG_OWN_CAR"]) > (np.minimum(((data["CNT_INSTALMENT_FUTURE"])), ((np.maximum(((data["te_FLAG_OWN_CAR"])), ((data["avg_buro_MONTHS_BALANCE"]))))))))*1.))))) 
    v["i406"] = np.tanh((-1.0*(((((((((data["CNT_INSTALMENT"]) - (np.where((13.12239170074462891)>0, np.where(data["NAME_YIELD_GROUP_low_normal"]>0, -1.0, data["AMT_GOODS_PRICE_x"] ), data["CNT_INSTALMENT"] )))) > (2.0))*1.)) / 2.0))))) 
    v["i407"] = np.tanh((((2.0) < (((data["CNT_INSTALMENT_FUTURE"]) - ((((data["AMT_GOODS_PRICE_x"]) < (2.0))*1.)))))*1.)) 
    v["i408"] = np.tanh((((3.0) < (((data["inst_AMT_PAYMENT"]) * (data["inst_AMT_PAYMENT"]))))*1.)) 
    v["i409"] = np.tanh(((np.tanh(((((2.0) < (np.minimum(((((((((data["CHANNEL_TYPE_Car_dealer"]) * 2.0)) * 2.0)) + ((((0.91963672637939453)) - (data["AMT_ANNUITY_x"])))))), (((6.00309991836547852))))))*1.)))) / 2.0)) 
    v["i410"] = np.tanh(np.tanh((np.where(np.tanh(((((data["inst_AMT_PAYMENT"]) < (data["inst_AMT_INSTALMENT"]))*1.)))>0, ((data["CHANNEL_TYPE_Car_dealer"]) - (data["inst_AMT_PAYMENT"])), ((data["NAME_GOODS_CATEGORY_Insurance"]) - (0.0)) )))) 
    v["i411"] = np.tanh((((((((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) / 2.0)) > (data["AMT_APPLICATION"]))*1.)) * ((((-1.0*((((data["AMT_APPLICATION"]) + ((-1.0*((data["AMT_CREDIT_y"]))))))))) * 2.0)))) * 2.0)) 
    v["i412"] = np.tanh(np.tanh((np.tanh((np.minimum((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) < (data["NAME_YIELD_GROUP_low_normal"]))*1.))), (((((data["AMT_CREDIT_y"]) > (data["AMT_APPLICATION"]))*1.))))))))) 
    v["i413"] = np.tanh((((-1.0*((((data["NAME_PRODUCT_TYPE_x_sell"]) * (data["inst_NUM_INSTALMENT_NUMBER"])))))) * (((((((np.maximum(((data["DAYS_FIRST_DRAWING"])), ((data["NAME_PRODUCT_TYPE_x_sell"])))) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) / 2.0)) * (1.0))))) 
    v["i414"] = np.tanh((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) > (np.maximum((((((((-2.0) < (data["DAYS_LAST_DUE_1ST_VERSION"]))*1.)) + (data["DAYS_LAST_DUE_1ST_VERSION"])))), ((((((data["DAYS_LAST_DUE_1ST_VERSION"]) + (data["DAYS_LAST_DUE_1ST_VERSION"]))) + (data["DAYS_LAST_DUE_1ST_VERSION"])))))))*1.)) 
    v["i415"] = np.tanh(((data["NAME_PRODUCT_TYPE_x_sell"]) * (np.where(((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) / 2.0)>0, data["NAME_CLIENT_TYPE_New"], ((np.minimum(((((data["NAME_CONTRACT_STATUS_Approved"]) + (2.0)))), ((((((8.13994503021240234)) < (3.0))*1.))))) * 2.0) )))) 
    v["i416"] = np.tanh(np.minimum(((((data["AMT_CREDIT_x"]) * ((((((((-1.0*((data["CHANNEL_TYPE_Car_dealer"])))) * 2.0)) * 2.0)) - (0.0)))))), (((-1.0*((data["CHANNEL_TYPE_Car_dealer"]))))))) 
    v["i417"] = np.tanh(((((((data["NAME_CONTRACT_STATUS_Approved"]) / 2.0)) * (data["PRODUCT_COMBINATION_Card_X_Sell"]))) * (np.minimum(((((data["DAYS_LAST_DUE_1ST_VERSION"]) * (data["PRODUCT_COMBINATION_Card_X_Sell"])))), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))))) 
    v["i418"] = np.tanh(np.maximum(((((data["inst_DAYS_ENTRY_PAYMENT"]) + (((-1.0) - ((((data["CODE_REJECT_REASON_HC"]) > (0.0))*1.))))))), ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (data["CODE_REJECT_REASON_HC"])))))) 
    v["i419"] = np.tanh(((((data["DAYS_FIRST_DUE"]) * (((((((data["DAYS_FIRST_DUE"]) * (data["cc_bal_AMT_PAYMENT_CURRENT"]))) / 2.0)) + ((-1.0*((np.tanh((np.tanh((data["PRODUCT_COMBINATION_Card_X_Sell"])))))))))))) / 2.0)) 
    v["i420"] = np.tanh((((np.minimum(((((((((1.0) * 2.0)) + (data["DAYS_FIRST_DUE"]))) * (data["DAYS_LAST_DUE"])))), ((np.tanh((0.0)))))) > (data["DAYS_TERMINATION"]))*1.)) 
    v["i421"] = np.tanh(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((data["DAYS_FIRST_DUE"]) + (((data["DAYS_FIRST_DUE"]) + (data["DAYS_TERMINATION"]))))))) * 2.0)) 
    v["i422"] = np.tanh((((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) > ((-1.0*((np.minimum(((data["NAME_CONTRACT_STATUS_Approved"])), (((((data["NAME_GOODS_CATEGORY_Insurance"]) + (data["SK_DPD_DEF"]))/2.0)))))))))*1.)) * 2.0)) * 2.0)) 
    v["i423"] = np.tanh(((np.where(data["PRODUCT_COMBINATION_Card_X_Sell"]>0, np.minimum(((np.tanh((data["AMT_CREDIT_y"])))), ((np.minimum((((-1.0*((data["avg_buro_buro_bal_status_1"]))))), ((data["SK_DPD_DEF"])))))), data["SK_DPD_DEF"] )) * 2.0)) 
    v["i424"] = np.tanh((-1.0*((np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, np.maximum(((data["AMT_GOODS_PRICE_x"])), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))), data["cc_bal_cc_bal_status__Sent_proposal"] ))))) 
    v["i425"] = np.tanh((((-1.0*((np.where((((13.84500789642333984)) * (data["CHANNEL_TYPE_Car_dealer"]))>0, data["OBS_60_CNT_SOCIAL_CIRCLE"], (-1.0*((np.maximum(((data["SK_DPD_DEF"])), ((((((3.0)) < (data["OBS_60_CNT_SOCIAL_CIRCLE"]))*1.))))))) ))))) * 2.0)) 
    v["i426"] = np.tanh((-1.0*(((((data["NAME_GOODS_CATEGORY_Insurance"]) + (((((np.minimum(((data["OBS_30_CNT_SOCIAL_CIRCLE"])), ((np.maximum(((((-1.0) + (data["NAME_YIELD_GROUP_high"])))), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))))) / 2.0)) + (data["SK_DPD_DEF"]))))/2.0))))) 
    v["i427"] = np.tanh(((((data["EXT_SOURCE_3"]) * (np.tanh(((((((((11.57633495330810547)) > (((data["SK_DPD_DEF"]) * ((5.0)))))*1.)) < (((np.minimum(((data["AMT_DOWN_PAYMENT"])), (((11.51627063751220703))))) * 2.0)))*1.)))))) * 2.0)) 
    v["i428"] = np.tanh(np.where(data["CNT_INSTALMENT"]>0, ((((data["SK_DPD_DEF"]) - ((((3.0) < (data["CHANNEL_TYPE_Car_dealer"]))*1.)))) * ((8.51310443878173828))), data["NAME_GOODS_CATEGORY_Insurance"] )) 
    v["i429"] = np.tanh(np.where(data["SK_DPD_DEF"]>0, ((np.minimum((((2.84313511848449707))), ((((((data["inst_DAYS_INSTALMENT"]) - (0.0))) * 2.0))))) * 2.0), (-1.0*((0.0))) )) 
    v["i430"] = np.tanh(((((np.minimum(((data["AMT_CREDIT_x"])), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))) + (data["SK_DPD_DEF"]))) * (np.where(data["SK_DPD_DEF"]>0, ((data["CNT_INSTALMENT_FUTURE"]) * 2.0), data["NAME_GOODS_CATEGORY_Insurance"] )))) 
    v["i431"] = np.tanh(((data["CNT_INSTALMENT_FUTURE"]) * (np.maximum(((np.minimum(((((1.0) - (data["SK_DPD_DEF"])))), ((data["SK_DPD_DEF"]))))), ((((data["SK_ID_PREV_x"]) / 2.0))))))) 
    v["i432"] = np.tanh(np.where(data["CHANNEL_TYPE_AP___Cash_loan_"]>0, (-1.0*((((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((((data["CNT_INSTALMENT"]) / 2.0))))) / 2.0)))), np.where(data["AMT_DOWN_PAYMENT"]>0, (((data["AMT_CREDIT_x"]) < (-1.0))*1.), data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"] ) )) 
    v["i433"] = np.tanh(np.minimum((((((((data["SK_DPD_DEF"]) * 2.0)) + (((np.tanh((data["REGION_POPULATION_RELATIVE"]))) / 2.0)))/2.0))), ((np.tanh((((data["SK_DPD_DEF"]) * (data["CNT_INSTALMENT"])))))))) 
    v["i434"] = np.tanh(((data["cnt_FLAG_DOCUMENT_10"]) + ((-1.0*((np.maximum(((np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, data["te_REGION_RATING_CLIENT_W_CITY"], data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"] ))), (((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) + (((data["SK_DPD_DEF"]) / 2.0)))/2.0)))))))))) 
    v["i435"] = np.tanh((((((-1.0*((np.where(np.where(((-1.0) / 2.0)>0, data["NAME_GOODS_CATEGORY_Insurance"], data["NAME_CASH_LOAN_PURPOSE_Business_development"] )>0, ((data["AMT_CREDIT_x"]) * 2.0), data["NAME_CASH_LOAN_PURPOSE_Business_development"] ))))) * 2.0)) * (data["REGION_RATING_CLIENT_W_CITY"]))) 
    v["i436"] = np.tanh(np.minimum((((-1.0*(((((data["AMT_CREDIT_x"]) > (((2.0) * 2.0)))*1.)))))), (((((((((-1.0*((data["cnt_FLAG_DOCUMENT_13"])))) < (np.tanh((data["cnt_REGION_RATING_CLIENT_W_CITY"]))))*1.)) + (data["AMT_CREDIT_x"]))/2.0))))) 
    v["i437"] = np.tanh(((((data["inst_AMT_PAYMENT"]) * 2.0)) * (np.where(((np.where(-3.0>0, data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"], data["cnt_NAME_HOUSING_TYPE"] )) - (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))>0, data["CHANNEL_TYPE_Car_dealer"], data["inst_AMT_PAYMENT"] )))) 
    v["i438"] = np.tanh(((-3.0) * ((((((((data["REGION_RATING_CLIENT_W_CITY"]) + (data["NAME_HOUSING_TYPE"]))) + (data["CHANNEL_TYPE_Car_dealer"]))) > (((((9.0)) + ((((data["CHANNEL_TYPE_Car_dealer"]) + (-1.0))/2.0)))/2.0)))*1.)))) 
    v["i439"] = np.tanh(((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * 2.0)) * ((((((-1.0*((data["te_REGION_RATING_CLIENT_W_CITY"])))) - ((((2.0) + (0.0))/2.0)))) - (data["REGION_RATING_CLIENT_W_CITY"]))))) 
    v["i440"] = np.tanh(((np.tanh((data["HOUR_APPR_PROCESS_START_x"]))) * ((((data["AMT_DOWN_PAYMENT"]) + (((((((data["AMT_CREDIT_x"]) < (2.0))*1.)) + (data["AMT_CREDIT_x"]))/2.0)))/2.0)))) 
    v["i441"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, (((data["cnt_FLAG_DOCUMENT_4"]) < ((-1.0*((data["te_HOUR_APPR_PROCESS_START_x"])))))*1.), (((-3.0) > (data["HOUR_APPR_PROCESS_START_y"]))*1.) )) 
    v["i442"] = np.tanh(((np.minimum(((np.minimum(((((data["HOUR_APPR_PROCESS_START_x"]) * (data["nans"])))), (((0.75266021490097046)))))), ((((((data["HOUR_APPR_PROCESS_START_x"]) * (np.tanh((data["NAME_GOODS_CATEGORY_Insurance"]))))) * 2.0))))) / 2.0)) 
    v["i443"] = np.tanh(((data["AMT_GOODS_PRICE_x"]) * (((data["FLAG_DOCUMENT_14"]) - ((((0.0) < (data["NAME_GOODS_CATEGORY_Fitness"]))*1.)))))) 
    v["i444"] = np.tanh(np.where(data["AMT_CREDIT_y"]>0, data["te_FLAG_DOCUMENT_14"], np.tanh(((-1.0*((np.where(3.0>0, ((((-1.0*((((data["AMT_CREDIT_x"]) / 2.0))))) + (0.0))/2.0), data["NAME_CASH_LOAN_PURPOSE_Business_development"] )))))) )) 
    v["i445"] = np.tanh((-1.0*(((-1.0*((((((data["te_FLAG_DOCUMENT_5"]) + (data["AMT_GOODS_PRICE_x"]))) * (data["FLAG_DOCUMENT_14"]))))))))) 
    v["i446"] = np.tanh(((np.minimum(((np.maximum(((data["FLAG_DOCUMENT_14"])), (((((data["AMT_GOODS_PRICE_x"]) < (data["te_FLAG_DOCUMENT_14"]))*1.)))))), ((data["cnt_FLAG_DOCUMENT_14"])))) * (((2.0) - (data["AMT_GOODS_PRICE_x"]))))) 
    v["i447"] = np.tanh(np.minimum((((((np.tanh(((-1.0*((((data["nans"]) / 2.0))))))) + (data["te_FLAG_DOCUMENT_14"]))/2.0))), ((0.0)))) 
    v["i448"] = np.tanh(np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])), (((((((((-2.0) - ((((data["FLAG_DOCUMENT_4"]) > (1.0))*1.)))) < ((((-1.0*((data["AMT_DOWN_PAYMENT"])))) * (2.0))))*1.)) + (data["AMT_CREDIT_y"])))))) 
    v["i449"] = np.tanh((((((data["CNT_INSTALMENT"]) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - (np.maximum(((data["AMT_CREDIT_x"])), ((((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))) + (1.0))/2.0))))))))) < (((((-2.0) / 2.0)) * 2.0)))*1.)) 
    v["i450"] = np.tanh((((data["AMT_DOWN_PAYMENT"]) > (((((2.39543843269348145)) + (((np.tanh((data["AMT_GOODS_PRICE_x"]))) * ((-1.0*(((((((data["AMT_GOODS_PRICE_x"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) < (data["AMT_DOWN_PAYMENT"]))*1.))))))))/2.0)))*1.)) 
    v["i451"] = np.tanh(((data["cnt_DAYS_BIRTH"]) * (((((data["CHANNEL_TYPE_Car_dealer"]) + (np.where(data["NAME_GOODS_CATEGORY_Fitness"]>0, data["AMT_GOODS_PRICE_x"], (((data["te_FLAG_DOCUMENT_20"]) + (data["NAME_GOODS_CATEGORY_Fitness"]))/2.0) )))) / 2.0)))) 
    v["i452"] = np.tanh(np.minimum(((data["SK_DPD_DEF"])), ((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["te_FLAG_DOCUMENT_20"], ((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (data["SK_DPD_DEF"]))) - (data["CHANNEL_TYPE_Car_dealer"]))) - (data["CHANNEL_TYPE_Car_dealer"])) ))))) 
    v["i453"] = np.tanh(np.minimum(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (data["AMT_GOODS_PRICE_x"])))), ((((0.0) - (((-2.0) + (np.maximum(((((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * 2.0)) * 2.0))), ((data["AMT_GOODS_PRICE_x"]))))))))))) 
    v["i454"] = np.tanh(((data["te_FLAG_CONT_MOBILE"]) * (np.where(2.0>0, np.where(((data["NAME_GOODS_CATEGORY_Insurance"]) + (data["AMT_CREDIT_y"]))>0, ((data["AMT_CREDIT_x"]) / 2.0), data["NAME_GOODS_CATEGORY_Fitness"] ), ((data["NAME_GOODS_CATEGORY_Fitness"]) - (data["te_FLAG_CONT_MOBILE"])) )))) 
    v["i455"] = np.tanh((((((data["te_FLAG_DOCUMENT_19"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) < (np.minimum(((((((data["te_FLAG_DOCUMENT_20"]) - (data["te_FLAG_EMAIL"]))) * 2.0))), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))))*1.)) 
    v["i456"] = np.tanh(((data["FLAG_DOCUMENT_15"]) * (np.minimum(((3.0)), ((np.minimum(((data["FLAG_DOCUMENT_15"])), ((((data["te_FLAG_DOCUMENT_20"]) - (((np.minimum((((5.04014062881469727))), (((((0.19776110351085663)) / 2.0))))) * 2.0)))))))))))) 
    v["i457"] = np.tanh(((np.minimum(((data["cnt_FLAG_DOCUMENT_4"])), ((((((1.0) - (data["te_FLAG_DOCUMENT_20"]))) - (np.tanh((np.tanh((data["NAME_GOODS_CATEGORY_Fitness"])))))))))) + (((data["FLAG_DOCUMENT_15"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))) 
    v["i458"] = np.tanh(np.where(data["FLAG_DOCUMENT_15"]>0, data["te_FLAG_DOCUMENT_20"], (((((-1.0*((data["te_FLAG_DOCUMENT_20"])))) * ((((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]) < (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))*1.)))) / 2.0) )) 
    v["i459"] = np.tanh(((data["FLAG_DOCUMENT_15"]) * (np.where(np.tanh((((data["FLAG_DOCUMENT_15"]) / 2.0)))>0, (((np.where(((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) / 2.0)>0, data["FLAG_DOCUMENT_15"], -2.0 )) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0), data["NAME_GOODS_CATEGORY_Fitness"] )))) 
    v["i460"] = np.tanh(((((((np.where(((((12.33196353912353516)) > (data["NAME_GOODS_CATEGORY_Fitness"]))*1.)>0, data["cnt_FLAG_DOCUMENT_10"], (((8.0)) * 2.0) )) * 2.0)) + (((np.tanh((data["cnt_FLAG_DOCUMENT_15"]))) - (data["NAME_GOODS_CATEGORY_Fitness"]))))) / 2.0)) 
    v["i461"] = np.tanh(((((np.minimum(((data["ty__Real_estate_loan"])), ((data["ty__Loan_for_working_capital_replenishment"])))) + (((data["AMT_CREDIT_y"]) * (data["ty__Loan_for_working_capital_replenishment"]))))) - (np.tanh((((((0.58618438243865967)) < (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))*1.)))))) 
    v["i462"] = np.tanh((((np.minimum(((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"])), ((np.where(np.where(-3.0>0, data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"], ((data["cc_bal_AMT_RECIVABLE"]) - (1.0)) )>0, data["NAME_GOODS_CATEGORY_Insurance"], ((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) * 2.0) ))))) > (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))*1.)) 
    v["i463"] = np.tanh((-1.0*((((np.where(data["AMT_CREDIT_y"]>0, data["AMT_GOODS_PRICE_x"], data["CNT_INSTALMENT_FUTURE"] )) * (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"])))))) 
    v["i464"] = np.tanh(((np.maximum(((np.where(data["ty__Loan_for_working_capital_replenishment"]>0, data["AMT_CREDIT_y"], data["ty__Loan_for_working_capital_replenishment"] ))), ((data["ty__Loan_for_working_capital_replenishment"])))) * (((np.tanh((data["AMT_CREDIT_y"]))) * 2.0)))) 
    v["i465"] = np.tanh(np.minimum(((np.tanh((((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]) * ((10.98657798767089844))))))), (((((-1.0*((((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))) < (np.minimum(((data["AMT_GOODS_PRICE_x"])), ((data["ty__Real_estate_loan"])))))*1.))))) * ((10.98657798767089844))))))) 
    v["i466"] = np.tanh(((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]) - (np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"])), ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - ((((((((data["ty__Interbank_credit"]) * (data["AMT_GOODS_PRICE_x"]))) * 2.0)) < (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * 2.0)))*1.))))))))) 
    v["i467"] = np.tanh(((((data["AMT_GOODS_PRICE_x"]) * (np.minimum(((((0.0) * 2.0))), (((((((data["NAME_GOODS_CATEGORY_House_Construction"]) > (0.0))*1.)) - (np.minimum(((data["nans"])), ((data["AMT_GOODS_PRICE_x"]))))))))))) * 2.0)) 
    v["i468"] = np.tanh(np.where(data["te_HOUR_APPR_PROCESS_START_x"]>0, ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * 2.0), ((np.tanh(((((((data["AMT_CREDIT_x"]) > ((((((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * 2.0)) * 2.0)) < (data["NAME_GOODS_CATEGORY_Direct_Sales"]))*1.)))*1.)) * 2.0)))) / 2.0) )) 
    v["i469"] = np.tanh(np.minimum(((((data["HOUR_APPR_PROCESS_START_x"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"])))), ((np.maximum((((-1.0*((np.where(1.0>0, data["HOUR_APPR_PROCESS_START_y"], data["HOUR_APPR_PROCESS_START_x"] )))))), ((np.tanh((data["HOUR_APPR_PROCESS_START_x"]))))))))) 
    v["i470"] = np.tanh((-1.0*((np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, data["EMERGENCYSTATE_MODE"], ((((((data["WALLSMATERIAL_MODE"]) * (data["HOUR_APPR_PROCESS_START_x"]))) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) * ((-1.0*((((data["HOUR_APPR_PROCESS_START_x"]) * 2.0)))))) ))))) 
    v["i471"] = np.tanh(np.minimum(((np.where(np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["NAME_GOODS_CATEGORY_Insurance"])))>0, data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"], data["cnt_FLAG_DOCUMENT_18"] ))), (((-1.0*(((((((data["te_HOUR_APPR_PROCESS_START_x"]) + (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) < (-1.0))*1.)))))))) 
    v["i472"] = np.tanh(((data["EXT_SOURCE_1"]) * (((((((1.0) * 2.0)) + (data["EXT_SOURCE_1"]))) - ((-1.0*((np.minimum(((((data["EXT_SOURCE_1"]) / 2.0))), ((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))))))))))) 
    v["i473"] = np.tanh((((10.0)) * (((np.minimum(((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * (data["EXT_SOURCE_1"])))), (((-1.0*((((((data["EXT_SOURCE_1"]) + ((((0.0) > (data["EXT_SOURCE_1"]))*1.)))) / 2.0)))))))) / 2.0)))) 
    v["i474"] = np.tanh(((np.maximum(((((data["te_DAYS_EMPLOYED"]) * ((0.0))))), ((np.tanh((np.where((8.0)>0, (((data["cnt_FLAG_DOCUMENT_3"]) > (data["te_ORGANIZATION_TYPE"]))*1.), 0.0 ))))))) / 2.0)) 
    v["i475"] = np.tanh(np.tanh((np.where(np.tanh((data["te_NAME_CONTRACT_TYPE"]))>0, np.where((((np.minimum(((data["te_NAME_CONTRACT_TYPE"])), ((-3.0)))) < (data["te_FLAG_DOCUMENT_8"]))*1.)>0, data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"], data["te_DAYS_EMPLOYED"] ), data["cnt_NAME_INCOME_TYPE"] )))) 
    v["i476"] = np.tanh(((((((data["te_ORGANIZATION_TYPE"]) * (np.tanh((data["AMT_CREDIT_x"]))))) / 2.0)) * (data["AMT_CREDIT_x"]))) 
    v["i477"] = np.tanh(((data["te_OCCUPATION_TYPE"]) * (np.minimum(((((data["cnt_FLAG_DOCUMENT_6"]) / 2.0))), (((-1.0*((np.minimum(((3.0)), ((np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, data["te_OCCUPATION_TYPE"], ((data["NAME_GOODS_CATEGORY_Insurance"]) / 2.0) ))))))))))))) 
    v["i478"] = np.tanh(((np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, (-1.0*((((3.0) - (data["cnt_FLAG_DOCUMENT_3"]))))), data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"] )) * ((((np.where(data["cnt_FLAG_DOCUMENT_3"]>0, 3.0, data["AMT_CREDIT_x"] )) < (data["cnt_FLAG_DOCUMENT_6"]))*1.)))) 
    v["i479"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["FLAG_DOCUMENT_6"], ((data["te_REGION_RATING_CLIENT"]) * ((-1.0*(((((np.minimum(((data["te_REGION_RATING_CLIENT"])), ((((data["te_FLAG_DOCUMENT_6"]) + (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))))) < (-2.0))*1.)))))) )) 
    v["i480"] = np.tanh(((np.where(data["cnt_FLAG_DOCUMENT_6"]>0, data["te_FLAG_DOCUMENT_6"], (-1.0*(((-1.0*((3.0)))))) )) * (((((((data["NAME_GOODS_CATEGORY_Insurance"]) < (data["te_REGION_RATING_CLIENT_W_CITY"]))*1.)) < (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)))) 
    v["i481"] = np.tanh(np.where(data["FLAG_DOCUMENT_6"]>0, data["CNT_FAM_MEMBERS"], ((((data["te_REGION_RATING_CLIENT_W_CITY"]) * (data["te_FLAG_DOCUMENT_6"]))) * (data["te_DAYS_EMPLOYED"])) )) 
    v["i482"] = np.tanh(np.where(((data["te_NAME_HOUSING_TYPE"]) - (1.0))>0, data["SK_ID_BUREAU"], np.maximum(((np.where(data["te_NAME_HOUSING_TYPE"]>0, 1.0, 0.0 ))), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) )) 
    v["i483"] = np.tanh(np.minimum(((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["te_NAME_CONTRACT_TYPE"]))) * 2.0))), ((((((((data["cnt_NAME_HOUSING_TYPE"]) > ((((data["NAME_CONTRACT_TYPE"]) + (((data["te_NAME_CONTRACT_TYPE"]) * (data["NAME_GOODS_CATEGORY_Insurance"]))))/2.0)))*1.)) + (data["NAME_CONTRACT_TYPE"]))/2.0))))) 
    v["i484"] = np.tanh(((np.where(((data["ty__Interbank_credit"]) * (data["NAME_GOODS_CATEGORY_Insurance"]))>0, data["NAME_GOODS_CATEGORY_Insurance"], np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((-1.0*((data["cnt_REGION_RATING_CLIENT_W_CITY"])))))) )) * ((((((data["NAME_GOODS_CATEGORY_Insurance"]) * 2.0)) > (data["NAME_GOODS_CATEGORY_Direct_Sales"]))*1.)))) 
    v["i485"] = np.tanh(np.where(((data["cnt_REGION_RATING_CLIENT"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))>0, (-1.0*((((data["cnt_REGION_RATING_CLIENT"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))))), (-1.0*((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_home"]) * 2.0)))) )) 
    v["i486"] = np.tanh(np.where(data["ty__Interbank_credit"]>0, ((data["NAME_GOODS_CATEGORY_Insurance"]) - (data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((((data["ty__Loan_for_working_capital_replenishment"]) * 2.0)) - (data["NAME_GOODS_CATEGORY_Insurance"])) )) 
    v["i487"] = np.tanh(((np.minimum(((((np.maximum(((((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) - (1.0)))), ((np.maximum(((data["ty__Loan_for_working_capital_replenishment"])), (((-1.0*((-3.0)))))))))) * (data["ty__Loan_for_working_capital_replenishment"])))), ((data["ty__Loan_for_working_capital_replenishment"])))) - (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) 
    v["i488"] = np.tanh((-1.0*(((((np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((np.where(data["ty__Loan_for_working_capital_replenishment"]>0, data["REGION_RATING_CLIENT_W_CITY"], 0.0 ))))) + ((((((data["DAYS_CREDIT"]) < (-3.0))*1.)) * 2.0)))/2.0))))) 
    v["i489"] = np.tanh(np.minimum(((np.where(data["cc_bal_CNT_DRAWINGS_CURRENT"]>0, data["REGION_RATING_CLIENT_W_CITY"], ((data["REGION_RATING_CLIENT_W_CITY"]) * (np.tanh((data["cc_bal_CNT_DRAWINGS_CURRENT"])))) ))), ((np.maximum(((-3.0)), ((0.0))))))) 
    v["i490"] = np.tanh(((np.maximum(((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"])))), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (np.minimum((((((-3.0) + (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))/2.0))), ((((-3.0) * 2.0)))))))))) * (data["te_REGION_RATING_CLIENT_W_CITY"]))) 
    v["i491"] = np.tanh((((np.where(np.tanh((data["NAME_GOODS_CATEGORY_Direct_Sales"]))>0, data["cc_bal_CNT_DRAWINGS_CURRENT"], ((((-2.0) + (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))) / 2.0) )) > (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))*1.)) 
    v["i492"] = np.tanh((((np.minimum(((data["ty__Loan_for_working_capital_replenishment"])), ((data["cc_bal_CNT_DRAWINGS_CURRENT"])))) > ((((np.tanh((np.tanh((data["cc_bal_CNT_DRAWINGS_CURRENT"]))))) < (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))*1.)))*1.)) 
    v["i493"] = np.tanh((((((np.where(np.where((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) < (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)>0, data["AMT_CREDIT_SUM_DEBT"], -3.0 )>0, data["NAME_GOODS_CATEGORY_Insurance"], -3.0 )) > (data["ty__Loan_for_working_capital_replenishment"]))*1.)) * 2.0)) 
    v["i494"] = np.tanh(np.tanh((np.where((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) + (np.tanh((-2.0))))/2.0)>0, data["cnt_REGION_RATING_CLIENT_W_CITY"], (((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) > (3.0))*1.) )))) 
    v["i495"] = np.tanh(np.where(data["DAYS_LAST_DUE_1ST_VERSION"]>0, ((data["inst_NUM_INSTALMENT_NUMBER"]) * (np.minimum(((np.where(data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]>0, -1.0, data["NAME_CASH_LOAN_PURPOSE_Business_development"] ))), ((data["NAME_GOODS_CATEGORY_Insurance"]))))), data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"] )) 
    v["i496"] = np.tanh((((((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) > (2.0))*1.)) + (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))) * ((((0.0) < ((((-1.0*((((2.0) - (data["cc_bal_AMT_TOTAL_RECEIVABLE"])))))) * 2.0)))*1.)))) 
    v["i497"] = np.tanh(((data["cc_bal_AMT_DRAWINGS_CURRENT"]) * (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - ((-1.0*(((-1.0*(((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) > (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + ((((2.0)) * 2.0)))))*1.)))))))))))) 
    v["i498"] = np.tanh(((((data["DAYS_LAST_DUE_1ST_VERSION"]) * (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))) * ((((((8.0)) * (data["ty__Loan_for_working_capital_replenishment"]))) - (data["ty__Loan_for_working_capital_replenishment"]))))) 
    v["i499"] = np.tanh((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) > (np.maximum(((np.where(3.0>0, ((data["cc_bal_AMT_PAYMENT_CURRENT"]) / 2.0), data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"] ))), ((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) * 2.0))))))*1.)) 
    v["i500"] = np.tanh((((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) * (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))) < ((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) + (np.minimum(((3.0)), ((-3.0)))))/2.0)))*1.)) * 2.0)) 
    v["i501"] = np.tanh(((((data["DAYS_LAST_DUE_1ST_VERSION"]) * ((2.0)))) * (np.where(data["ty__Loan_for_working_capital_replenishment"]>0, data["cc_bal_AMT_PAYMENT_CURRENT"], ((np.tanh((data["DAYS_LAST_DUE_1ST_VERSION"]))) * (data["NAME_GOODS_CATEGORY_Direct_Sales"])) )))) 
    v["i502"] = np.tanh(np.where(((((((6.0)) * 2.0)) > (((data["ty__Loan_for_working_capital_replenishment"]) - ((-1.0*((data["FLAG_DOCUMENT_4"])))))))*1.)>0, np.maximum(((data["ty__Interbank_credit"])), ((data["ty__Loan_for_working_capital_replenishment"]))), -2.0 )) 
    v["i503"] = np.tanh(np.where(data["ty__Loan_for_working_capital_replenishment"]>0, (((2.0) < (data["AMT_CREDIT_x"]))*1.), np.tanh((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, (((2.0) < (data["AMT_CREDIT_x"]))*1.), 0.0 ))) )) 
    v["i504"] = np.tanh((-1.0*((np.where(((((((data["nans"]) + (1.0))/2.0)) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))/2.0)>0, data["NAME_GOODS_CATEGORY_Insurance"], data["AMT_GOODS_PRICE_x"] ))))) 
    v["i505"] = np.tanh(((data["ELEVATORS_MODE"]) * (((np.where(((np.where((6.0)>0, data["cc_bal_CNT_INSTALMENT_MATURE_CUM"], np.maximum(((0.0)), (((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))))) )) / 2.0)>0, data["cc_bal_CNT_INSTALMENT_MATURE_CUM"], data["ty__Loan_for_working_capital_replenishment"] )) * 2.0)))) 
    v["i506"] = np.tanh(((((2.09928321838378906)) < (np.maximum(((((data["cc_bal_SK_ID_PREV"]) * (-2.0)))), ((np.where(data["AMT_CREDIT_x"]>0, data["cc_bal_CNT_INSTALMENT_MATURE_CUM"], np.tanh((1.0)) ))))))*1.)) 
    v["i507"] = np.tanh(((((data["AMT_CREDIT_x"]) * (((data["cc_bal_SK_ID_PREV"]) * (data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]))))) / 2.0)) 
    v["i508"] = np.tanh((-1.0*((np.maximum((((((np.tanh((((((((data["ty__Loan_for_working_capital_replenishment"]) * (data["nans"]))) / 2.0)) + (data["ty__Interbank_credit"]))))) < (data["ty__Loan_for_working_capital_replenishment"]))*1.))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"]))))))) 
    v["i509"] = np.tanh(np.where(((np.tanh((np.maximum(((((((data["ty__Interbank_credit"]) * (data["AMT_GOODS_PRICE_x"]))) - (data["te_HOUSETYPE_MODE"])))), ((data["cnt_WALLSMATERIAL_MODE"])))))) / 2.0)>0, ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (data["AMT_GOODS_PRICE_x"])), data["AMT_GOODS_PRICE_x"] )) 
    v["i510"] = np.tanh(np.minimum(((0.0)), (((((((data["te_HOUSETYPE_MODE"]) < ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) < (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))*1.)))*1.)) - ((((data["cnt_WALLSMATERIAL_MODE"]) + (data["AMT_GOODS_PRICE_x"]))/2.0))))))) 
    v["i511"] = np.tanh(((np.maximum(((((data["te_FLAG_DOCUMENT_7"]) + (-1.0)))), (((((-1.0*((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["te_FLAG_DOCUMENT_10"])))))) * (-1.0)))))) * 2.0))

    return v

gc.enable()

buro_bal = pd.read_csv('../input/bureau_balance.csv.zip')
print('Buro bal shape : ', buro_bal.shape)

print('transform to dummies')
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)

print('Counting buros')
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

print('averaging buro bal')
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
del buro_bal
gc.collect()

print('Read Bureau')
buro = pd.read_csv('../input/bureau.csv.zip')

print('Go to dummies')
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
# buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
gc.collect()

print('Merge with buro avg')
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

print('Counting buro per SK_ID_CURR')
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

print('Averaging bureau')
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
print(avg_buro.head())

del buro, buro_full
gc.collect()

print('Read prev')
prev = pd.read_csv('../input/previous_application.csv.zip')

prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]

print('Go to dummies')
prev_dum = pd.DataFrame()
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

prev = pd.concat([prev, prev_dum], axis=1)

del prev_dum
gc.collect()

print('Counting number of Prevs')
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

print('Averaging prev')
avg_prev = prev.groupby('SK_ID_CURR').mean()
print(avg_prev.head())
del prev
gc.collect()

print('Reading POS_CASH')
pos = pd.read_csv('../input/POS_CASH_balance.csv.zip')

print('Go to dummies')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

print('Compute nb of prevs per curr')
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Go to averages')
avg_pos = pos.groupby('SK_ID_CURR').mean()

del pos, nb_prevs
gc.collect()

print('Reading CC balance')
cc_bal = pd.read_csv('../input/credit_card_balance.csv.zip')

print('Go to dummies')
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Compute average')
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

del cc_bal, nb_prevs
gc.collect()

print('Reading Installments')
inst = pd.read_csv('../input/installments_payments.csv.zip')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

print('Read data and test')
data = pd.read_csv('../input/application_train.csv.zip')
test = pd.read_csv('../input/application_test.csv.zip')
print('Shapes : ', data.shape, test.shape)


print('Read data and test')
train = pd.read_csv('../input/application_train.csv.zip')
test = pd.read_csv('../input/application_test.csv.zip')
print('Shapes : ', train.shape, train.shape)

categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    train[f_], indexer = pd.factorize(train[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
train = train.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

del avg_buro, avg_prev
gc.collect()

ID = test.SK_ID_CURR

train.columns = train.columns.str.replace('[^A-Za-z0-9_]', '_')
test.columns = test.columns.str.replace('[^A-Za-z0-9_]', '_')

floattypes = []
inttypes = []
stringtypes = []
for c in test.columns:
    if(train[c].dtype=='object'):
        train[c] = train[c].astype('str')
        stringtypes.append(c)
    elif(train[c].dtype=='int64'):
        train[c] = train[c].astype('int32')
        inttypes.append(c)
    else:
        train[c] = train[c].astype('float32')
        floattypes.append(c)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in stringtypes:
    train['te_'+col] = 0.
    test['te_'+col] = 0.
    SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]
    _, test['te_'+col] = target_encode(train[col], 
                                      test[col], 
                                      target=train['TARGET'], 
                                      min_samples_leaf=100,
                                      smoothing=SMOOTHING,
                                      noise_level=0.0)
    for f, (vis_index, blind_index) in enumerate(kf.split(train)):
        _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                            train.loc[blind_index, col], 
                                                            target=train.loc[vis_index,'TARGET'], 
                                                            min_samples_leaf=100,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in inttypes:
    train['te_'+col] = 0.
    test['te_'+col] = 0.
    SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]
    _, test['te_'+col] = target_encode(train[col], 
                                      test[col], 
                                      target=train['TARGET'], 
                                      min_samples_leaf=100,
                                      smoothing=SMOOTHING,
                                      noise_level=0.0)
    for f, (vis_index, blind_index) in enumerate(kf.split(train)):
        _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                            train.loc[blind_index, col], 
                                                            target=train.loc[vis_index,'TARGET'], 
                                                            min_samples_leaf=100,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.0)

ntrainrows = train.shape[0]
alldata = pd.concat([train,test])
del train ,test
gc.collect()

alldata['nans'] = alldata.isnull().sum(axis=1)

for col in inttypes[1:]:
    x = alldata[col].value_counts().reset_index(drop=False)
    x.columns = [col,'cnt_'+col]
    x['cnt_'+col]/=alldata.shape[0]
    alldata = alldata.merge(x,on=col,how='left')
    
features = list(set(alldata.columns).difference(['SK_ID_CURR','TARGET']))
alldata[features] = alldata[features].astype('float32')


for c in features:
    ss = StandardScaler()
    alldata.loc[~alldata[c].isnull(),c] = ss.fit_transform(alldata.loc[~alldata[c].isnull(),c].values.reshape(-1,1))
    alldata[c].fillna(alldata[c].mean(),inplace=True)


train = alldata[:ntrainrows]
test = alldata[ntrainrows:]

traintargets = train.TARGET.values
train = UseGPFeatures(train)
test = UseGPFeatures(test)


# =============================================================================
# output
# =============================================================================
train.reset_index(drop=True).to_feather('../data/train_gp1.f')
test.reset_index(drop=True).to_feather('../data/test_gp1.f')


#==============================================================================


