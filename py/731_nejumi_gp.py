#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:42:51 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import utils
utils.start(__file__)
#==============================================================================

def GP1(data):
    v = pd.DataFrame()
    v["i6"] = 0.099849*np.tanh(((data["nejumi"]) + (((((((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["NAME_INCOME_TYPE_Working"]) + (((data["nejumi"]) + (((data["REG_CITY_NOT_WORK_CITY"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))))))))) - (data["CODE_GENDER"]))) * 2.0)))) 
    v["i8"] = 0.099495*np.tanh(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, (5.0), np.where(data["nejumi"]>0, data["nejumi"], np.where(data["nejumi"]<0, ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) + (data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]))) * ((8.84948539733886719))), (13.10789012908935547) ) ) )) 
    v["i12"] = 0.099924*np.tanh((-1.0*((np.where(np.minimum(((data["NAME_FAMILY_STATUS_Married"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"])))>0, (10.0), (((6.0)) * (np.where(data["NEW_DOC_IND_KURT"]<0, 3.0, ((data["NEW_DOC_IND_KURT"]) + (data["nejumi"])) ))) ))))) 
    v["i18"] = 0.099955*np.tanh(np.where(((data["nejumi"]) + (((data["CLOSED_DAYS_CREDIT_VAR"]) * (data["NAME_FAMILY_STATUS_Married"])))) < -99998, data["ORGANIZATION_TYPE_Self_employed"], ((data["ORGANIZATION_TYPE_Self_employed"]) - ((((5.0)) * (((1.0) + (data["nejumi"])))))) )) 
    v["i20"] = 0.099957*np.tanh(((np.tanh((data["REFUSED_CNT_PAYMENT_SUM"]))) + (np.maximum((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["FLAG_WORK_PHONE"]))/2.0))), ((((np.maximum(((data["nejumi"])), ((np.maximum(((data["ORGANIZATION_TYPE_Transport__type_3"])), ((data["DAYS_ID_PUBLISH"]))))))) - (0.636620)))))))) 
    v["i31"] = 0.099942*np.tanh((((5.04678153991699219)) * (np.where(((data["nejumi"]) + (data["ORGANIZATION_TYPE_Realtor"])) < -99998, data["OCCUPATION_TYPE_Cleaning_staff"], (((-1.0*((np.tanh((np.tanh((0.318310)))))))) - (data["nejumi"])) )))) 
    v["i62"] = 0.100000*np.tanh(((((((-1.0) / 2.0)) - (np.where(data["nejumi"] < -99998, 0.0, ((data["nejumi"]) * 2.0) )))) - ((((np.tanh((((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) / 2.0)))) < (data["nejumi"]))*1.)))) 
    v["i64"] = 0.099952*np.tanh((-1.0*((((np.where(data["nejumi"] < -99998, np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]<0, 0.318310, data["nejumi"] ), ((data["nejumi"]) - (np.tanh((((np.tanh((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * 2.0))))) )) * 2.0))))) 
    v["i65"] = 0.099900*np.tanh(np.where(data["nejumi"] < -99998, 0.318310, (-1.0*((((((data["nejumi"]) - ((-1.0*(((((((data["nejumi"]) > (-1.0))*1.)) - (data["ORGANIZATION_TYPE_XNA"])))))))) * ((8.0)))))) )) 
    v["i67"] = 0.099733*np.tanh((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.)) - (np.where(data["nejumi"] < -99998, ((((1.0)) > (data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]))*1.), data["nejumi"] )))) - ((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)))) 
    v["i69"] = 0.099999*np.tanh(np.where((((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) + ((((3.0) > (data["nejumi"]))*1.)))/2.0)<0, np.where(data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"]>0, (8.0), data["INSTAL_DAYS_ENTRY_PAYMENT_SUM"] ), ((((data["INSTAL_AMT_INSTALMENT_MAX"]) * 2.0)) * 2.0) )) 
    v["i70"] = 0.099998*np.tanh(((((data["nejumi"]) / 2.0)) - (np.minimum((((((np.tanh((data["ACTIVE_MONTHS_BALANCE_MAX_MAX"]))) + (3.141593))/2.0))), ((np.maximum(((((((((data["nejumi"]) * 2.0)) * 2.0)) * 2.0))), ((data["nejumi"]))))))))) 
    v["i71"] = 0.099949*np.tanh(np.where((((data["nejumi"]) + (np.tanh((1.0))))/2.0)<0, np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (12.80127429962158203) ), (-1.0*(((12.80127429962158203)))) )) 
    v["i72"] = 0.099500*np.tanh((((((data["nejumi"]) > (((-2.0) * 2.0)))*1.)) * ((((((((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) < (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))*1.)) - (0.636620))) - (data["nejumi"]))) * 2.0)))) 
    v["i74"] = 0.099260*np.tanh((((((-1.0*(((((data["nejumi"]) > (((np.minimum((((((data["nejumi"]) < (((0.636620) * 2.0)))*1.))), ((0.636620)))) + (np.tanh((0.318310))))))*1.))))) * 2.0)) * 2.0)) 
    v["i75"] = 0.099504*np.tanh((-1.0*((np.where(np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]<0, data["nejumi"], 0.318310 ) < -99998, 0.318310, np.where(data["nejumi"] < -99998, data["nejumi"], np.where(data["nejumi"]<0, data["nejumi"], data["nejumi"] ) ) ))))) 
    v["i78"] = 0.099301*np.tanh(np.minimum(((((1.0) - (data["DAYS_BIRTH"])))), ((np.where(data["OCCUPATION_TYPE_Laborers"]>0, np.maximum(((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"])), ((data["DAYS_BIRTH"]))), (((((data["nejumi"]) / 2.0)) < (data["NEW_INC_PER_CHLD"]))*1.) ))))) 
    v["i92"] = 0.098000*np.tanh(np.minimum(((np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_State_servant"], np.where((((1.0) + (data["nejumi"]))/2.0)>0, data["nejumi"], (8.0) ) ))), ((((0.636620) - (data["nejumi"])))))) 
    v["i97"] = 0.076491*np.tanh(np.where(np.where(data["nejumi"]>0, data["DAYS_BIRTH"], data["ENTRANCES_MODE"] )>0, (((((data["DAYS_BIRTH"]) < (data["ENTRANCES_MODE"]))*1.)) - (data["DAYS_BIRTH"])), ((np.tanh((np.tanh((data["DAYS_BIRTH"]))))) / 2.0) )) 
    v["i99"] = 0.071179*np.tanh(np.where(data["nejumi"] < -99998, (-1.0*(((((((0.0) > (data["AMT_INCOME_TOTAL"]))*1.)) / 2.0)))), ((np.tanh((np.where(data["nejumi"]>0, 0.636620, data["nejumi"] )))) - (data["nejumi"])) )) 
    v["i105"] = 0.095005*np.tanh(np.where((((data["nejumi"]) + (-1.0))/2.0)<0, np.maximum(((((data["WALLSMATERIAL_MODE_Stone__brick"]) * (data["AMT_INCOME_TOTAL"])))), ((data["NAME_INCOME_TYPE_Maternity_leave"]))), ((((data["CLOSED_AMT_ANNUITY_MEAN"]) + (data["WALLSMATERIAL_MODE_Stone__brick"]))) * 2.0) )) 
    v["i106"] = 0.096790*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]<0, ((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"] )) * 2.0), np.where(data["nejumi"]<0, ((np.maximum(((data["NAME_INCOME_TYPE_Maternity_leave"])), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"])))) * 2.0), data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] ) )) 
    v["i108"] = 0.098660*np.tanh((((((data["nejumi"]) - (((1.570796) - (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))))) > (np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]<0, data["ACTIVE_DAYS_CREDIT_VAR"], 3.0 ), ((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) * 2.0) )))*1.)) 
    v["i116"] = 0.099950*np.tanh(np.where(data["nejumi"]<0, np.where(data["nejumi"] < -99998, np.where(((0.636620) + (data["nejumi"]))<0, data["NAME_INCOME_TYPE_Maternity_leave"], data["nejumi"] ), data["nejumi"] ), ((2.0) + (data["nejumi"])) )) 
    v["i118"] = 0.099801*np.tanh(((((((((((np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], (((data["nejumi"]) < (((((0.318310) / 2.0)) - (2.0))))*1.) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i124"] = 0.099896*np.tanh(np.where(((data["nejumi"]) - (-1.0))<0, (((((-1.0) * 2.0)) < (data["nejumi"]))*1.), (((((((data["NEW_INC_PER_CHLD"]) * 2.0)) < (data["nejumi"]))*1.)) * (data["CC_AMT_DRAWINGS_OTHER_CURRENT_MAX"])) )) 
    v["i128"] = 0.096500*np.tanh(((np.where(data["FLAG_EMAIL"]<0, (-1.0*((np.where(data["NAME_INCOME_TYPE_Maternity_leave"]<0, ((3.0) * ((((data["nejumi"]) > (1.570796))*1.))), data["nejumi"] )))), data["nejumi"] )) - (data["NAME_INCOME_TYPE_Student"]))) 
    v["i129"] = 0.100000*np.tanh((((((np.maximum(((data["BURO_STATUS_X_MEAN_MEAN"])), ((np.maximum(((data["nejumi"])), ((((((-1.0*((0.318310)))) > (data["nejumi"]))*1.)))))))) < (0.318310))*1.)) - ((((1.570796) < (data["nejumi"]))*1.)))) 
    v["i147"] = 0.099951*np.tanh(((np.tanh((((((((data["AMT_ANNUITY"]) * ((13.30679225921630859)))) * 2.0)) * 2.0)))) - (np.where(data["NAME_INCOME_TYPE_Student"] < -99998, data["nejumi"], ((((data["nejumi"]) + (data["nejumi"]))) * 2.0) )))) 
    v["i149"] = 0.093000*np.tanh(np.where((((1.570796) + (np.where(data["nejumi"] < -99998, data["nejumi"], data["nejumi"] )))/2.0)<0, data["ORGANIZATION_TYPE_Advertising"], np.where((((data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]) + (1.570796))/2.0)<0, data["nejumi"], data["nejumi"] ) )) 
    v["i150"] = 0.096966*np.tanh((((np.minimum(((data["nejumi"])), ((np.where(data["HOUR_APPR_PROCESS_START"]>0, data["nejumi"], data["NAME_INCOME_TYPE_Maternity_leave"] ))))) > ((((((0.318310) * (np.minimum(((-2.0)), ((data["nejumi"])))))) + (data["nejumi"]))/2.0)))*1.)) 
    v["i156"] = 0.099200*np.tanh(((((data["ORGANIZATION_TYPE_Realtor"]) + (3.0))) * (((data["ORGANIZATION_TYPE_Realtor"]) + ((((data["ORGANIZATION_TYPE_Construction"]) > (np.where(data["nejumi"] < -99998, data["ORGANIZATION_TYPE_Construction"], (((data["nejumi"]) + (1.570796))/2.0) )))*1.)))))) 
    v["i158"] = 0.082050*np.tanh(((data["nejumi"]) * ((((((((data["nejumi"]) > (((((data["NAME_TYPE_SUITE_Group_of_people"]) + (3.0))) - (0.318310))))*1.)) - (data["NAME_TYPE_SUITE_Group_of_people"]))) - (0.318310))))) 
    v["i163"] = 0.099819*np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MAX"]>0, (((data["AMT_INCOME_TOTAL"]) > (0.318310))*1.), (((-2.0) > (np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["nejumi"]) - (data["NAME_INCOME_TYPE_Maternity_leave"])) )))*1.) )) 
    v["i191"] = 0.099503*np.tanh((-1.0*((((((((((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"] < -99998, data["nejumi"], 1.570796 )) + (data["AMT_ANNUITY"]))/2.0)) > (2.0))*1.)) * (np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, (5.0), -2.0 ))))))) 
    v["i192"] = 0.099943*np.tanh(np.where(((0.636620) - (data["nejumi"]))<0, data["OCCUPATION_TYPE_Accountants"], np.where(data["nejumi"] < -99998, np.where(data["AMT_INCOME_TOTAL"]<0, data["AMT_INCOME_TOTAL"], (((data["AMT_ANNUITY"]) < (data["ORGANIZATION_TYPE_Advertising"]))*1.) ), data["ORGANIZATION_TYPE_Advertising"] ) )) 
    v["i197"] = 0.098240*np.tanh(((((((data["WALLSMATERIAL_MODE_Others"]) + (np.where(data["nejumi"]>0, np.maximum(((data["ORGANIZATION_TYPE_Housing"])), ((data["WALLSMATERIAL_MODE_Others"]))), (((data["ORGANIZATION_TYPE_Housing"]) < (data["nejumi"]))*1.) )))) * (((data["AMT_ANNUITY"]) * 2.0)))) / 2.0)) 
    v["i200"] = 0.079602*np.tanh(np.minimum(((((1.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))), ((((((((data["AMT_ANNUITY"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) < (((((((data["AMT_ANNUITY"]) > (data["nejumi"]))*1.)) < (((data["AMT_ANNUITY"]) * 2.0)))*1.)))*1.))))) 
    v["i204"] = 0.098990*np.tanh(np.where(data["CLOSED_AMT_ANNUITY_MEAN"] < -99998, (((-1.0*(((((((data["nejumi"]) < (((((0.636620) / 2.0)) + (data["AMT_ANNUITY"]))))*1.)) / 2.0))))) / 2.0), (((0.636620) < (data["AMT_ANNUITY"]))*1.) )) 
    v["i264"] = 0.099459*np.tanh(((((((np.maximum(((data["nejumi"])), ((((2.0) - (data["nejumi"])))))) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))/2.0)) < ((((((2.0) - (data["nejumi"]))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)))*1.)) 
    v["i280"] = 0.097030*np.tanh(np.minimum(((((data["REGION_POPULATION_RELATIVE"]) * (((data["nejumi"]) / 2.0))))), ((((data["AMT_CREDIT"]) * (np.where(((data["nejumi"]) / 2.0)<0, data["NAME_INCOME_TYPE_Maternity_leave"], ((data["REGION_POPULATION_RELATIVE"]) * (data["nejumi"])) ))))))) 
    v["i282"] = 0.098896*np.tanh(((np.where(data["nejumi"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where((((data["nejumi"]) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)>0, data["nejumi"], (((np.maximum(((data["NEW_CREDIT_TO_ANNUITY_RATIO"])), ((1.570796)))) < (data["HOUR_APPR_PROCESS_START"]))*1.) ) )) * 2.0)) 
    v["i283"] = 0.099500*np.tanh(np.maximum(((((data["nejumi"]) * ((((3.141593) < (data["nejumi"]))*1.))))), (((((np.tanh((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"]))))) > (((np.maximum(((data["AMT_INCOME_TOTAL"])), ((0.318310)))) / 2.0)))*1.))))) 
    v["i332"] = 0.099999*np.tanh(((((((data["NEW_INC_PER_CHLD"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, data["OCCUPATION_TYPE_Low_skill_Laborers"], data["nejumi"] )))*1.)))) * (2.0))) * ((((-1.0) < (data["nejumi"]))*1.)))) 
    v["i378"] = 0.097398*np.tanh((((((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + ((((((data["NAME_TYPE_SUITE_Other_B"]) * (data["NAME_TYPE_SUITE_Other_B"]))) < (data["nejumi"]))*1.)))/2.0)<0, data["NAME_INCOME_TYPE_Student"], 1.570796 )) < (data["nejumi"]))*1.)) * (-2.0))) 
    v["i379"] = 0.099776*np.tanh((((-1.0*((data["OCCUPATION_TYPE_Laborers"])))) * ((((((((((np.where(data["nejumi"] < -99998, (1.77269029617309570), (((data["OCCUPATION_TYPE_Laborers"]) < (2.0))*1.) )) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.)) * 2.0)) * 2.0)) * 2.0)))) 
    v["i388"] = 0.099710*np.tanh(np.maximum(((np.tanh((data["NEW_CAR_TO_EMPLOY_RATIO"])))), ((np.minimum((((((((data["NAME_INCOME_TYPE_Pensioner"]) + (data["nejumi"]))/2.0)) * (((((data["NAME_INCOME_TYPE_Pensioner"]) * 2.0)) * 2.0))))), ((np.maximum(((0.0)), ((data["NAME_INCOME_TYPE_Pensioner"])))))))))) 
    v["i390"] = 0.100000*np.tanh((-1.0*((((np.where(((((((1.0) + (1.570796))/2.0)) + (data["nejumi"]))/2.0)>0, data["DAYS_BIRTH"], (((-1.0*((0.318310)))) * (data["DAYS_BIRTH"])) )) * 2.0))))) 
    v["i392"] = 0.090389*np.tanh(np.where(data["nejumi"]>0, ((data["nejumi"]) - (data["nejumi"])), (((data["nejumi"]) > (np.tanh((((((data["CLOSED_AMT_CREDIT_SUM_MEAN"]) - (1.0))) - (data["nejumi"]))))))*1.) )) 
    v["i395"] = 0.099623*np.tanh(np.where((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > (3.141593))*1.)>0, (-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))), ((np.tanh((np.tanh((np.tanh((((data["nejumi"]) * ((-1.0*((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))))))))))) / 2.0) )) 
    v["i401"] = 0.089300*np.tanh((((((data["nejumi"]) > (np.where(data["nejumi"]>0, (((data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]) < (data["AMT_ANNUITY"]))*1.), data["AMT_ANNUITY"] )))*1.)) * ((((((data["NAME_INCOME_TYPE_Student"]) > (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) + (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))))) 
    v["i408"] = 0.086000*np.tanh((((((6.0)) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))) * ((-1.0*(((((np.where(data["nejumi"] < -99998, ((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) - ((-1.0*((data["AMT_REQ_CREDIT_BUREAU_MON"]))))), data["nejumi"] )) > ((1.48802196979522705)))*1.))))))) 
    v["i409"] = 0.092999*np.tanh(np.maximum((((-1.0*((((1.0) / 2.0)))))), ((np.where((((data["nejumi"]) + (2.0))/2.0)>0, (-1.0*((((data["nejumi"]) - ((-1.0*((data["DAYS_BIRTH"])))))))), data["DAYS_BIRTH"] ))))) 
    v["i411"] = 0.097800*np.tanh(np.where(((data["nejumi"]) + (data["nejumi"]))<0, np.where(data["nejumi"] < -99998, 0.0, (((data["DAYS_BIRTH"]) < (np.tanh(((((data["nejumi"]) < (data["NAME_INCOME_TYPE_Student"]))*1.)))))*1.) ), data["AMT_ANNUITY"] )) 
    return v.add_prefix('gp1_')

def GP2(data):
    v = pd.DataFrame()
    v["i5"] = 0.099985*np.tanh(((((np.maximum(((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"])), ((((((np.where(data["nejumi"]<0, data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"], data["NEW_DOC_IND_KURT"] )) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) + (np.maximum(((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"])), ((data["nejumi"]))))))))) * 2.0)) * 2.0)) 
    v["i10"] = 0.100000*np.tanh(((((((data["nejumi"]) - (np.where((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + ((((data["CLOSED_AMT_CREDIT_SUM_SUM"]) + (np.tanh((1.0))))/2.0)))/2.0)>0, (1.0), data["NAME_FAMILY_STATUS_Married"] )))) * 2.0)) - (data["CLOSED_AMT_CREDIT_SUM_MEAN"]))) 
    v["i11"] = 0.099915*np.tanh((((((((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (data["nejumi"]))*1.)) - (((np.where(data["nejumi"] < -99998, np.tanh((data["ACTIVE_DAYS_CREDIT_VAR"])), data["nejumi"] )) - (np.tanh((data["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"]))))))) * 2.0)) * 2.0)) 
    v["i13"] = 0.100000*np.tanh((((((((-1.0*((((((np.maximum(((data["nejumi"])), ((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])))) * 2.0)) * 2.0))))) - (np.maximum(((((data["OCCUPATION_TYPE_Core_staff"]) / 2.0))), ((0.318310)))))) * 2.0)) * 2.0)) 
    v["i15"] = 0.099997*np.tanh(((((((data["OCCUPATION_TYPE_Low_skill_Laborers"]) + (data["REG_CITY_NOT_LIVE_CITY"]))) - (data["FLAG_PHONE"]))) + (np.where(data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["DAYS_REGISTRATION"], ((data["ORGANIZATION_TYPE_Self_employed"]) + (((data["nejumi"]) + (data["REG_CITY_NOT_LIVE_CITY"])))) )))) 
    v["i18"] = 0.099990*np.tanh((-1.0*((np.where(((np.maximum(((np.maximum(((data["OCCUPATION_TYPE_Accountants"])), ((data["nejumi"]))))), ((np.maximum(((data["NEW_CAR_TO_EMPLOY_RATIO"])), ((data["NEW_CAR_TO_EMPLOY_RATIO"]))))))) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))<0, data["nejumi"], (13.81985569000244141) ))))) 
    v["i34"] = 0.099997*np.tanh(np.minimum(((((np.maximum(((data["nejumi"])), ((data["DAYS_REGISTRATION"])))) / 2.0))), ((((((((0.636620) - ((-1.0*((((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (0.636620))) * 2.0))))))) * 2.0)) * 2.0))))) 
    v["i64"] = 0.099960*np.tanh(((((((np.where(((((data["nejumi"]) + (data["nejumi"]))) + (1.570796))>0, -2.0, np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (10.0) ) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i65"] = 0.099910*np.tanh(np.where(((data["nejumi"]) - (np.tanh((((data["nejumi"]) * 2.0)))))>0, -2.0, (-1.0*((np.where(data["nejumi"] < -99998, data["NAME_EDUCATION_TYPE_Academic_degree"], ((((data["nejumi"]) * 2.0)) * 2.0) )))) )) 
    v["i66"] = 0.099640*np.tanh(np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], ((((((((-1.0) - (((data["nejumi"]) * 2.0)))) - (data["nejumi"]))) * 2.0)) - (np.minimum(((data["nejumi"])), ((data["nejumi"]))))) )) 
    v["i67"] = 0.099950*np.tanh((-1.0*(((((((data["nejumi"]) < (((data["REFUSED_AMT_GOODS_PRICE_MAX"]) + (1.570796))))*1.)) + (np.where(data["nejumi"] < -99998, np.tanh((-2.0)), ((data["nejumi"]) * 2.0) ))))))) 
    v["i68"] = 0.099490*np.tanh((-1.0*((np.maximum(((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"])), ((np.where(data["nejumi"]>0, data["DAYS_BIRTH"], (-1.0*(((((((((((((data["DAYS_BIRTH"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)) > (data["NAME_INCOME_TYPE_Maternity_leave"]))*1.)))) )))))))) 
    v["i75"] = 0.099510*np.tanh(((np.minimum(((((1.570796) - (np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["nejumi"] ))))), ((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, data["NAME_INCOME_TYPE_Unemployed"], data["nejumi"] )) - (data["nejumi"])))))) * 2.0)) 
    v["i76"] = 0.098997*np.tanh(np.where(data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"] < -99998, ((-1.0) / 2.0), np.where(data["CLOSED_DAYS_CREDIT_MEAN"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) * 2.0), np.where(data["nejumi"] < -99998, ((data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]) * 2.0), ((2.0) * 2.0) ) ) )) 
    v["i77"] = 0.099680*np.tanh(np.tanh((np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, np.where(data["REFUSED_APP_CREDIT_PERC_MAX"] < -99998, np.where(data["nejumi"]<0, 1.0, data["nejumi"] ), data["REFUSED_AMT_GOODS_PRICE_MAX"] ), (-1.0*((data["nejumi"]))) )))) 
    v["i79"] = 0.099996*np.tanh(((np.where(data["nejumi"] < -99998, ((((data["ORGANIZATION_TYPE_Legal_Services"]) * 2.0)) * 2.0), (-1.0*(((((((((data["nejumi"]) * 2.0)) + (1.570796))/2.0)) + (((data["nejumi"]) * 2.0)))))) )) * 2.0)) 
    v["i82"] = 0.099796*np.tanh(np.where(data["nejumi"] < -99998, ((-1.0) / 2.0), ((((((np.minimum(((data["nejumi"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))) * (data["nejumi"]))) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))) + (np.tanh((-1.0)))) )) 
    v["i85"] = 0.099980*np.tanh(((((((np.tanh((-1.0))) - (data["nejumi"]))) - (data["CLOSED_AMT_CREDIT_SUM_DEBT_SUM"]))) * (((-2.0) - (((np.where(data["nejumi"] < -99998, -1.0, data["nejumi"] )) * 2.0)))))) 
    v["i86"] = 0.099797*np.tanh(np.where(data["nejumi"] < -99998, ((np.where(data["ACTIVE_DAYS_CREDIT_VAR"]<0, data["DAYS_BIRTH"], data["ACTIVE_DAYS_CREDIT_VAR"] )) * 2.0), (-1.0*((((((((data["nejumi"]) + (data["DAYS_BIRTH"]))) * 2.0)) + (data["DAYS_BIRTH"]))))) )) 
    v["i89"] = 0.099620*np.tanh(np.where(data["POS_SK_DPD_DEF_MAX"]>0, (-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))), np.where(data["nejumi"] < -99998, np.minimum(((data["POS_SK_DPD_DEF_MAX"])), ((-1.0))), np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, (-1.0*((data["nejumi"]))), -1.0 ) ) )) 
    v["i93"] = 0.098996*np.tanh(((np.where(np.where(data["nejumi"] < -99998, ((-1.0) + (data["nejumi"])), (-1.0*((data["ORGANIZATION_TYPE_Bank"]))) )>0, data["nejumi"], data["ORGANIZATION_TYPE_Bank"] )) * (((-1.0) + (data["nejumi"]))))) 
    v["i94"] = 0.099400*np.tanh(np.tanh((np.maximum(((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]<0, ((data["nejumi"]) - (2.0)), data["nejumi"] ))), ((((np.where(data["NAME_EDUCATION_TYPE_Higher_education"]<0, data["NAME_INCOME_TYPE_State_servant"], data["CLOSED_DAYS_CREDIT_VAR"] )) - (data["nejumi"])))))))) 
    v["i97"] = 0.097551*np.tanh(((np.where(data["nejumi"]>0, data["NAME_CONTRACT_TYPE_Cash_loans"], (((((-2.0) + ((((((data["NAME_INCOME_TYPE_Unemployed"]) > (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) * 2.0)))) + ((((data["NAME_CONTRACT_TYPE_Cash_loans"]) < (data["ORGANIZATION_TYPE_Legal_Services"]))*1.)))/2.0) )) * 2.0)) 
    v["i105"] = 0.099940*np.tanh(((np.where(data["nejumi"]<0, data["FONDKAPREMONT_MODE_reg_oper_spec_account"], ((((((((data["FONDKAPREMONT_MODE_reg_oper_spec_account"]) + (1.0))/2.0)) + (data["NEW_CREDIT_TO_INCOME_RATIO"]))/2.0)) - (((data["nejumi"]) * 2.0))) )) - (((data["nejumi"]) * 2.0)))) 
    v["i107"] = 0.092699*np.tanh((((((data["REGION_RATING_CLIENT"]) < ((((((data["nejumi"]) < (-1.0))*1.)) - ((-1.0*((((np.minimum(((((data["REGION_RATING_CLIENT"]) * (data["nejumi"])))), ((data["nejumi"])))) / 2.0))))))))*1.)) * 2.0)) 
    v["i119"] = 0.093920*np.tanh(((np.minimum(((data["nejumi"])), ((np.where(np.where(data["ORGANIZATION_TYPE_Legal_Services"]>0, 0.636620, data["nejumi"] )>0, 0.636620, (-1.0*((1.570796))) ))))) - (data["nejumi"]))) 
    v["i121"] = 0.096997*np.tanh(np.minimum((((((data["nejumi"]) < (1.0))*1.))), ((np.where(data["nejumi"]>0, data["NEW_INC_PER_CHLD"], (((1.570796) < (np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], (-1.0*((data["nejumi"]))) )))*1.) ))))) 
    v["i123"] = 0.100000*np.tanh(((((np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], (((((-1.0) - (data["nejumi"]))) > (0.636620))*1.) )) * 2.0)) + ((-1.0*(((((1.0) < (data["nejumi"]))*1.))))))) 
    v["i132"] = 0.099999*np.tanh((((((((np.maximum(((1.570796)), ((data["nejumi"])))) < (data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.)) + (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, (((3.141593) < (data["nejumi"]))*1.), np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"])) )))) * 2.0)) 
    v["i133"] = 0.099840*np.tanh((((((data["DAYS_BIRTH"]) * (data["nejumi"]))) + (np.minimum(((((((data["DAYS_BIRTH"]) * 2.0)) * ((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))))), ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))))/2.0)) 
    v["i134"] = 0.099021*np.tanh(((np.where(((((((3.0) - (1.570796))) - (data["NAME_INCOME_TYPE_Student"]))) - (data["DAYS_BIRTH"]))>0, ((data["NAME_EDUCATION_TYPE_Academic_degree"]) * (((data["nejumi"]) * 2.0))), -2.0 )) * 2.0)) 
    v["i148"] = 0.099000*np.tanh(np.minimum((((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) < (data["nejumi"]))*1.))), ((((((np.where(data["nejumi"]>0, data["NEW_DOC_IND_KURT"], (((((1.0) + (data["NEW_DOC_IND_KURT"]))/2.0)) - (data["nejumi"])) )) * 2.0)) * 2.0))))) 
    v["i152"] = 0.034897*np.tanh(np.where(data["NEW_DOC_IND_KURT"]>0, ((np.where((-1.0*((data["nejumi"])))>0, (-1.0*((0.636620))), data["AMT_CREDIT"] )) / 2.0), np.where(data["NAME_INCOME_TYPE_Student"]<0, (-1.0*((data["nejumi"]))), data["NEW_DOC_IND_KURT"] ) )) 
    v["i159"] = 0.099450*np.tanh(((((data["NEW_DOC_IND_KURT"]) / 2.0)) * (np.where(((0.318310) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))>0, np.where(data["nejumi"]<0, ((data["AMT_ANNUITY"]) / 2.0), data["nejumi"] ), data["nejumi"] )))) 
    v["i162"] = 0.099900*np.tanh(((((data["nejumi"]) * (((((np.tanh((np.tanh((np.minimum(((((data["nejumi"]) * 2.0))), ((data["AMT_ANNUITY"])))))))) - (data["nejumi"]))) * 2.0)))) * (((data["nejumi"]) * 2.0)))) 
    v["i213"] = 0.099993*np.tanh(np.where(data["BURO_AMT_CREDIT_SUM_MEAN"] < -99998, ((data["nejumi"]) - (data["AMT_REQ_CREDIT_BUREAU_WEEK"])), ((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) - (3.141593))) * (data["BURO_AMT_CREDIT_SUM_MEAN"]))) * ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (data["BURO_AMT_CREDIT_SUM_MEAN"]))*1.))) )) 
    v["i214"] = 0.099000*np.tanh(np.where((((-2.0) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)>0, 1.570796, (-1.0*(((((data["nejumi"]) > (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (1.570796))*1.), 1.570796 )))*1.)))) )) 
    v["i215"] = 0.099496*np.tanh(np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Maternity_leave"], np.where((((2.0) < (((data["NAME_INCOME_TYPE_Maternity_leave"]) - (data["nejumi"]))))*1.)>0, (((4.0)) * 2.0), ((data["AMT_REQ_CREDIT_BUREAU_DAY"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"])) ) )) 
    v["i216"] = 0.052589*np.tanh(((((((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - ((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))/2.0)))) * ((-1.0*(((((data["nejumi"]) > ((((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (0.0))*1.)))*1.))))))) * 2.0)) * 2.0)) 
    v["i220"] = 0.096999*np.tanh((-1.0*((np.where(((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) - ((((data["nejumi"]) < (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.)))>0, data["nejumi"], ((((-1.0*((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"])))) < ((((data["ORGANIZATION_TYPE_Advertising"]) < (data["ACTIVE_AMT_ANNUITY_MAX"]))*1.)))*1.) ))))) 
    v["i245"] = 0.098999*np.tanh(((data["nejumi"]) * ((-1.0*((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, ((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) + ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.))), (((data["nejumi"]) > (1.570796))*1.) ))))))) 
    v["i246"] = 0.076350*np.tanh(((((data["nejumi"]) * 2.0)) * ((((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, ((((-1.0*((data["nejumi"])))) > (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.), (-1.0*((data["AMT_ANNUITY"]))) )) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))) 
    v["i247"] = 0.099104*np.tanh((((((np.tanh((data["nejumi"]))) > ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + (data["nejumi"]))/2.0)))*1.)) - ((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) < ((-1.0*((data["nejumi"])))))*1.)) + (data["nejumi"]))))) 
    v["i248"] = 0.090002*np.tanh(((((data["DAYS_BIRTH"]) + ((((-1.0) + (data["BURO_CREDIT_TYPE_Car_loan_MEAN"]))/2.0)))) * (np.minimum(((((data["BURO_CREDIT_TYPE_Consumer_credit_MEAN"]) - (data["BURO_CREDIT_TYPE_Car_loan_MEAN"])))), (((((((data["nejumi"]) * 2.0)) > (data["DAYS_BIRTH"]))*1.))))))) 
    v["i249"] = 0.099966*np.tanh((-1.0*((((data["nejumi"]) - ((((((((np.tanh((((np.tanh((((((data["DAYS_BIRTH"]) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * (data["CODE_GENDER"]))) + (-1.0))/2.0))))))) 
    v["i251"] = 0.099990*np.tanh((-1.0*((np.where((-1.0*((data["nejumi"])))>0, data["REGION_RATING_CLIENT"], ((data["nejumi"]) + ((((data["nejumi"]) > ((((((data["nejumi"]) + (0.636620))) > (data["REGION_RATING_CLIENT"]))*1.)))*1.))) ))))) 
    v["i253"] = 0.099500*np.tanh(((((data["nejumi"]) + ((13.18767356872558594)))) * ((((np.where(data["nejumi"]>0, ((((((data["nejumi"]) + (-1.0))/2.0)) < (data["CLOSED_MONTHS_BALANCE_SIZE_SUM"]))*1.), -2.0 )) > (data["nejumi"]))*1.)))) 
    v["i254"] = 0.099742*np.tanh(((((3.0) - (np.where(np.where(data["nejumi"] < -99998, ((3.0) - (data["ACTIVE_AMT_ANNUITY_MEAN"])), data["DAYS_BIRTH"] )<0, data["nejumi"], 3.141593 )))) * 2.0)) 
    v["i255"] = 0.099548*np.tanh(np.where(data["nejumi"] < -99998, np.where(data["nejumi"] < -99998, data["nejumi"], ((np.tanh((np.tanh((data["nejumi"]))))) - (data["nejumi"])) ), np.minimum(((0.636620)), ((data["nejumi"]))) )) 
    v["i256"] = 0.089995*np.tanh(np.where(data["AMT_REQ_CREDIT_BUREAU_QRT"]>0, np.minimum(((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))), ((data["AMT_REQ_CREDIT_BUREAU_QRT"]))), (((data["nejumi"]) > (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) - (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), ((-1.0)))))))*1.) )) 
    v["i257"] = 0.099999*np.tanh(((np.tanh((np.minimum(((((np.where(data["nejumi"] < -99998, data["HOUR_APPR_PROCESS_START"], np.maximum(((data["NEW_INC_BY_ORG"])), ((data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"]))) )) + (data["HOUR_APPR_PROCESS_START"])))), ((((data["nejumi"]) * (data["NEW_INC_BY_ORG"])))))))) / 2.0)) 
    v["i259"] = 0.044980*np.tanh(np.maximum(((np.where(data["LIVINGAREA_MODE"]<0, data["OCCUPATION_TYPE_Medicine_staff"], (((np.where(data["DAYS_ID_PUBLISH"]<0, 1.570796, data["OCCUPATION_TYPE_High_skill_tech_staff"] )) > (data["OCCUPATION_TYPE_Medicine_staff"]))*1.) ))), ((((-2.0) - (data["nejumi"])))))) 
    v["i261"] = 0.099990*np.tanh((((((((data["nejumi"]) < (((np.tanh((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))) - (((((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) > (((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) - (data["CLOSED_MONTHS_BALANCE_MAX_MAX"]))))*1.)) + (3.141593))/2.0)))))*1.)) * 2.0)) * 2.0)) 
    v["i262"] = 0.094001*np.tanh(np.minimum((((((data["nejumi"]) > (0.636620))*1.))), (((((11.55127620697021484)) * ((((((((0.636620) > (((data["nejumi"]) / 2.0)))*1.)) / 2.0)) - (((data["nejumi"]) / 2.0))))))))) 
    v["i264"] = 0.096890*np.tanh(((np.minimum(((np.minimum(((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0))), ((data["nejumi"]))))), ((data["AMT_INCOME_TOTAL"])))) * ((-1.0*(((((np.where(data["NEW_CAR_TO_EMPLOY_RATIO"]<0, data["AMT_INCOME_TOTAL"], 3.141593 )) + (data["NAME_EDUCATION_TYPE_Higher_education"]))/2.0))))))) 
    v["i266"] = 0.083000*np.tanh(np.maximum((((((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (data["nejumi"]))*1.)) * (data["nejumi"])))), ((np.where(data["nejumi"]>0, np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["nejumi"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] ), ((data["nejumi"]) * 2.0) ))))) 
    v["i268"] = 0.097000*np.tanh(((((((np.where(data["nejumi"] < -99998, data["WEEKDAY_APPR_PROCESS_START_TUESDAY"], (((0.636620) > (((3.0) + (((data["nejumi"]) + (data["NAME_INCOME_TYPE_Maternity_leave"]))))))*1.) )) * 2.0)) * 2.0)) * 2.0)) 
    v["i270"] = 0.099960*np.tanh(np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]>0, (((data["nejumi"]) < (-1.0))*1.), ((np.where(data["nejumi"]<0, data["ORGANIZATION_TYPE_Trade__type_3"], (-1.0*((data["ORGANIZATION_TYPE_Trade__type_3"]))) )) - ((((1.570796) < (data["nejumi"]))*1.))) )) 
    v["i271"] = 0.096900*np.tanh(np.where(data["BURO_AMT_ANNUITY_MAX"]<0, np.where(data["ORGANIZATION_TYPE_Trade__type_3"]<0, (((0.636620) < (((data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]) / 2.0)))*1.), ((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) * 2.0)) * 2.0) ), (((data["BURO_AMT_ANNUITY_MAX"]) > (data["nejumi"]))*1.) )) 
    v["i272"] = 0.056000*np.tanh((((((((((-1.0*((np.maximum(((np.where(data["nejumi"]<0, (-1.0*((np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((data["NAME_EDUCATION_TYPE_Academic_degree"])))))), data["APPROVED_AMT_DOWN_PAYMENT_MEAN"] ))), ((data["NAME_EDUCATION_TYPE_Academic_degree"]))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i273"] = 0.095900*np.tanh(np.maximum((((((((1.570796) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) * 2.0)))*1.)) * ((-1.0*((data["nejumi"]))))))), ((np.where(((data["nejumi"]) - (1.570796))>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0 ))))) 
    v["i274"] = 0.065300*np.tanh((((((((((((data["NEW_CAR_TO_BIRTH_RATIO"]) < ((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["OWN_CAR_AGE"]))*1.)))*1.)) > (data["nejumi"]))*1.)) < (((data["OWN_CAR_AGE"]) * (((1.570796) / 2.0)))))*1.)) * (1.570796))) 
    v["i275"] = 0.098970*np.tanh(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((0.636620) < (((data["NEW_CAR_TO_BIRTH_RATIO"]) - (((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["NEW_CAR_TO_BIRTH_RATIO"]) < (((data["nejumi"]) * (data["nejumi"]))))*1.)))))))*1.)))) 
    v["i277"] = 0.099600*np.tanh(np.where(data["nejumi"] < -99998, data["nejumi"], ((((((-1.0) < (np.where(data["OWN_CAR_AGE"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["NEW_CAR_TO_BIRTH_RATIO"]))) )))*1.)) < (((data["nejumi"]) / 2.0)))*1.) )) 
    v["i278"] = 0.098097*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((3.0) * (np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (3.0))*1.)>0, data["nejumi"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (((data["nejumi"]) + (3.141593))))*1.) )))))) 
    v["i281"] = 0.091999*np.tanh((((((((data["AMT_ANNUITY"]) > ((2.73518514633178711)))*1.)) * 2.0)) + (((np.where(data["nejumi"] < -99998, data["nejumi"], (((data["nejumi"]) < ((-1.0*(((2.73518514633178711))))))*1.) )) + (data["ORGANIZATION_TYPE_Telecom"]))))) 
    v["i297"] = 0.098995*np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (np.minimum(((np.where(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"]))>0, data["ORGANIZATION_TYPE_Advertising"], data["AMT_ANNUITY"] ))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["AMT_ANNUITY"]))) - (data["nejumi"])))))))) 
    v["i299"] = 0.099500*np.tanh((((((-2.0) > (np.where(data["nejumi"] < -99998, ((-2.0) * (data["AMT_ANNUITY"])), ((np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), (((((-1.0*((data["nejumi"])))) * 2.0))))) * 2.0) )))*1.)) * 2.0)) 
    v["i301"] = 0.057002*np.tanh(((((((((data["BURO_DAYS_CREDIT_VAR"]) / 2.0)) < (data["nejumi"]))*1.)) < (((((-1.0*((0.318310)))) < (np.minimum(((((data["BURO_DAYS_CREDIT_VAR"]) * 2.0))), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])))))*1.)))*1.)) 
    v["i302"] = 0.094999*np.tanh((((((((-1.0*((np.maximum(((data["CLOSED_DAYS_CREDIT_VAR"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))))) < (((1.0) - (np.minimum(((0.636620)), ((1.0)))))))*1.)) < ((((data["nejumi"]) > (data["BURO_DAYS_CREDIT_VAR"]))*1.)))*1.)) 
    v["i341"] = 0.090000*np.tanh(np.where(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]>0, data["AMT_CREDIT"], (((data["nejumi"]) > (data["AMT_CREDIT"]))*1.) )>0, 0.318310, (-1.0*(((((data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((((data["NAME_INCOME_TYPE_Student"]) > (data["AMT_CREDIT"]))*1.)))*1.)))) )) 
    v["i427"] = 0.099750*np.tanh(np.where(((data["nejumi"]) - (data["NAME_INCOME_TYPE_Student"]))<0, data["AMT_INCOME_TOTAL"], (((((-1.0*((data["nejumi"])))) * (((data["nejumi"]) - ((((data["nejumi"]) < (1.0))*1.)))))) * 2.0) )) 
    v["i432"] = 0.091453*np.tanh((((((2.0) < (np.where(data["NAME_INCOME_TYPE_State_servant"]<0, (-1.0*((((data["nejumi"]) - ((-1.0*((0.318310)))))))), data["REGION_POPULATION_RELATIVE"] )))*1.)) * 2.0)) 
    v["i433"] = 0.095979*np.tanh(np.where(((data["NAME_FAMILY_STATUS_Civil_marriage"]) - (((data["nejumi"]) + ((2.0)))))<0, (((np.where(data["nejumi"]<0, data["nejumi"], data["NAME_FAMILY_STATUS_Civil_marriage"] )) > (data["ORGANIZATION_TYPE_Military"]))*1.), data["NAME_INCOME_TYPE_State_servant"] )) 
    v["i434"] = 0.054620*np.tanh(np.where(data["CC_AMT_PAYMENT_CURRENT_VAR"] < -99998, ((np.tanh((((data["nejumi"]) * 2.0)))) * (np.minimum(((data["REGION_RATING_CLIENT"])), ((0.318310))))), np.minimum(((data["nejumi"])), ((((data["REGION_RATING_CLIENT"]) + (data["REGION_POPULATION_RELATIVE"]))))) )) 
    v["i437"] = 0.088430*np.tanh(np.where(np.where(data["NAME_TYPE_SUITE_Group_of_people"]<0, data["ORGANIZATION_TYPE_Trade__type_7"], data["nejumi"] )<0, (((data["nejumi"]) > (1.570796))*1.), np.where(data["nejumi"]<0, data["ORGANIZATION_TYPE_Trade__type_7"], -2.0 ) )) 
    v["i438"] = 0.013965*np.tanh(np.where(data["nejumi"]>0, ((np.tanh((np.tanh((data["NEW_ANNUITY_TO_INCOME_RATIO"]))))) / 2.0), (-1.0*((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((((((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) / 2.0)) / 2.0)) / 2.0))))))) )) 
    v["i439"] = 0.097690*np.tanh((-1.0*(((((-1.0) > (np.maximum((((((((data["nejumi"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) + (((-1.0) * 2.0)))/2.0))), ((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) + (0.636620)))))))*1.))))) 
    v["i441"] = 0.099940*np.tanh(np.maximum((((((np.tanh((1.570796))) < (((data["NEW_CAR_TO_EMPLOY_RATIO"]) - ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["nejumi"]))/2.0)))))*1.))), (((((((data["AMT_CREDIT"]) < (data["nejumi"]))*1.)) * (data["AMT_CREDIT"])))))) 
    v["i442"] = 0.020019*np.tanh((((np.where(((-1.0) + (data["AMT_CREDIT"]))<0, data["OWN_CAR_AGE"], ((np.where(data["nejumi"]<0, ((-1.0) + (data["AMT_CREDIT"])), -1.0 )) * 2.0) )) > ((1.32165348529815674)))*1.)) 
    v["i444"] = 0.069500*np.tanh((((((((((3.0) - (data["AMT_CREDIT"]))) < ((((np.where(data["nejumi"]>0, 0.318310, ((3.0) - (0.318310)) )) + (data["BURO_STATUS_5_MEAN_MEAN"]))/2.0)))*1.)) * 2.0)) * 2.0)) 
    v["i445"] = 0.099749*np.tanh(np.where(((data["NEW_CAR_TO_EMPLOY_RATIO"]) * ((((data["AMT_CREDIT"]) < (data["nejumi"]))*1.)))>0, data["nejumi"], ((data["AMT_CREDIT"]) * (np.minimum(((data["NAME_EDUCATION_TYPE_Lower_secondary"])), ((((data["NAME_INCOME_TYPE_Student"]) - (data["NAME_EDUCATION_TYPE_Lower_secondary"]))))))) )) 
    v["i446"] = 0.033999*np.tanh((-1.0*(((((((data["NEW_CAR_TO_BIRTH_RATIO"]) > (data["NEW_CAR_TO_EMPLOY_RATIO"]))*1.)) * (np.maximum(((np.where(data["OWN_CAR_AGE"]<0, ((data["NEW_CAR_TO_BIRTH_RATIO"]) * (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"])), (-1.0*((data["NEW_CREDIT_TO_INCOME_RATIO"]))) ))), ((data["nejumi"]))))))))) 
    v["i447"] = 0.099649*np.tanh(((((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) - (-2.0))) + (((3.0) / 2.0)))) * ((((data["nejumi"]) > ((((2.36968207359313965)) / 2.0)))*1.)))) 
    v["i450"] = 0.099984*np.tanh((((data["nejumi"]) > (((((((3.0) + (data["ORGANIZATION_TYPE_Industry__type_5"]))) + (data["ORGANIZATION_TYPE_Trade__type_4"]))) - (np.where(((data["nejumi"]) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))>0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_Industry__type_1"] )))))*1.)) 
    v["i451"] = 0.097200*np.tanh(np.where(data["nejumi"]<0, data["ORGANIZATION_TYPE_Advertising"], np.where(((data["AMT_ANNUITY"]) * (data["NEW_ANNUITY_TO_INCOME_RATIO"]))<0, -1.0, (((data["nejumi"]) < ((((data["ORGANIZATION_TYPE_Industry__type_5"]) + (data["AMT_ANNUITY"]))/2.0)))*1.) ) )) 
    v["i452"] = 0.099052*np.tanh((-1.0*((np.where(np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"]>0, data["NEW_CAR_TO_EMPLOY_RATIO"], data["nejumi"] ) < -99998, 1.570796, np.where(data["BURO_CREDIT_TYPE_Another_type_of_loan_MEAN"]<0, np.where(data["BURO_CNT_CREDIT_PROLONG_SUM"]>0, -2.0, data["ORGANIZATION_TYPE_Industry__type_5"] ), 3.141593 ) ))))) 
    v["i483"] = 0.099230*np.tanh((((7.0)) * ((((data["BURO_DAYS_CREDIT_VAR"]) > (np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]<0, np.where(data["NAME_FAMILY_STATUS_Single___not_married"]<0, np.where(data["nejumi"] < -99998, data["NAME_FAMILY_STATUS_Single___not_married"], (7.0) ), data["NAME_FAMILY_STATUS_Single___not_married"] ), (2.46100592613220215) )))*1.)))) 
    v["i484"] = 0.099970*np.tanh((((((1.570796) < (data["BURO_DAYS_CREDIT_VAR"]))*1.)) * (np.where(np.where(data["AMT_INCOME_TOTAL"]>0, data["nejumi"], data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] )>0, np.maximum(((data["nejumi"])), ((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))), (-1.0*((1.570796))) )))) 
    v["i488"] = 0.069600*np.tanh(((((((1.0)) < (((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"]) * 2.0)))*1.)) * (np.where(data["nejumi"]<0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]>0, -2.0, np.maximum(((data["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"])), (((3.15189313888549805)))) ) )))) 
    v["i489"] = 0.098019*np.tanh((-1.0*(((((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((-1.0*((np.where(data["nejumi"] < -99998, data["FLAG_EMP_PHONE"], (((data["NAME_FAMILY_STATUS_Single___not_married"]) + ((-1.0*((((data["nejumi"]) / 2.0))))))/2.0) ))))))*1.))))) 
    v["i492"] = 0.056000*np.tanh((((((((((1.570796) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)) * (((((data["nejumi"]) * 2.0)) * 2.0)))) + (((((((data["nejumi"]) < (1.570796))*1.)) < (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)))) * 2.0)) 
    v["i495"] = 0.099459*np.tanh(((data["DAYS_BIRTH"]) * ((-1.0*((np.where(data["nejumi"]<0, ((data["DAYS_BIRTH"]) * ((-1.0*(((((data["DAYS_BIRTH"]) < (data["nejumi"]))*1.)))))), np.maximum(((data["nejumi"])), ((data["DAYS_BIRTH"]))) ))))))) 
    v["i496"] = 0.021039*np.tanh(np.where(data["nejumi"]<0, data["OCCUPATION_TYPE_Secretaries"], (-1.0*(((((((data["nejumi"]) > ((((np.tanh((data["nejumi"]))) > ((((data["nejumi"]) + (((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) * 2.0)))/2.0)))*1.)))*1.)) * 2.0)))) )) 
    v["i497"] = 0.099830*np.tanh((((data["OCCUPATION_TYPE_Laborers"]) < (((((np.minimum(((data["NAME_INCOME_TYPE_Student"])), ((((((data["nejumi"]) * 2.0)) - (data["nejumi"])))))) * 2.0)) - (((data["nejumi"]) / 2.0)))))*1.)) 
    v["i498"] = 0.038590*np.tanh(((np.where(data["ORGANIZATION_TYPE_Military"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (-1.0*(((((((np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((data["nejumi"])))) + (np.maximum(((data["ORGANIZATION_TYPE_Military"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))) > (2.0))*1.)))) )) * 2.0)) 
    v["i499"] = 0.046598*np.tanh(((((((data["nejumi"]) < (np.minimum((((((data["OCCUPATION_TYPE_Laborers"]) < (data["nejumi"]))*1.))), ((data["nejumi"])))))*1.)) + ((-1.0*(((((-2.0) < (np.minimum(((data["nejumi"])), ((data["nejumi"])))))*1.))))))/2.0)) 
    v["i500"] = 0.099900*np.tanh((((((data["nejumi"]) > (np.where(data["nejumi"]<0, data["NAME_INCOME_TYPE_Student"], np.maximum((((3.0))), ((data["AMT_ANNUITY"]))) )))*1.)) * (np.where(data["AMT_ANNUITY"]>0, (4.22247409820556641), data["nejumi"] )))) 
    v["i501"] = 0.095040*np.tanh((((((((3.0) < (np.maximum(((data["nejumi"])), ((np.where(np.maximum(((data["AMT_ANNUITY"])), ((data["OCCUPATION_TYPE_Laborers"])))<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"])), data["AMT_INCOME_TOTAL"] ))))))*1.)) * 2.0)) * 2.0)) 
    v["i503"] = 0.099960*np.tanh(((np.where((((data["nejumi"]) + ((((0.318310) + (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))/2.0)))/2.0)<0, 0.0, (-1.0*(((((data["APPROVED_AMT_DOWN_PAYMENT_MAX"]) > (np.maximum(((data["nejumi"])), ((data["NAME_INCOME_TYPE_Student"])))))*1.)))) )) * 2.0)) 
    v["i504"] = 0.000009*np.tanh(((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) - ((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) > (np.maximum(((data["APPROVED_AMT_DOWN_PAYMENT_MAX"])), ((data["APPROVED_RATE_DOWN_PAYMENT_MAX"])))))*1.)))) * ((((data["nejumi"]) > ((((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))) < (data["APPROVED_AMT_DOWN_PAYMENT_MAX"]))*1.)))*1.)))) 
    v["i505"] = 0.014603*np.tanh((((((np.tanh((data["nejumi"]))) > ((((data["PREV_AMT_DOWN_PAYMENT_MAX"]) > (np.tanh((np.tanh((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]))))))*1.)))*1.)) * (((((data["APPROVED_RATE_DOWN_PAYMENT_MAX"]) + (-1.0))) + (-1.0))))) 
    v["i506"] = 0.064800*np.tanh(((((((data["nejumi"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) * 2.0)))*1.)) < ((((((((((((data["APPROVED_RATE_DOWN_PAYMENT_MEAN"]) < (((data["PREV_AMT_DOWN_PAYMENT_MEAN"]) / 2.0)))*1.)) > (data["nejumi"]))*1.)) / 2.0)) + (data["PREV_AMT_DOWN_PAYMENT_MAX"]))/2.0)))*1.)) 
    v["i507"] = 0.099096*np.tanh((((((np.where(data["nejumi"] < -99998, data["AMT_ANNUITY"], data["nejumi"] )) < (data["nejumi"]))*1.)) * (np.where(data["nejumi"]>0, data["AMT_ANNUITY"], np.minimum(((data["nejumi"])), (((-1.0*((data["nejumi"])))))) )))) 
    v["i508"] = 0.052010*np.tanh((-1.0*(((((((np.where(data["nejumi"] < -99998, 3.141593, ((data["nejumi"]) * (((((data["nejumi"]) - (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))) - (data["NAME_INCOME_TYPE_Student"])))) )) < (data["APPROVED_AMT_DOWN_PAYMENT_MEAN"]))*1.)) * 2.0))))) 
    v["i509"] = 0.099609*np.tanh((-1.0*(((((data["nejumi"]) > (np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, (((data["nejumi"]) > (np.tanh((0.636620))))*1.), np.where(np.tanh((data["nejumi"]))<0, 0.318310, data["AMT_ANNUITY"] ) )))*1.))))) 
    return v.add_prefix('gp2_')


def GP3(data):
    v = pd.DataFrame()
    v["i5"] = 0.099954*np.tanh(((((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, 1.570796, data["nejumi"] )) + (((np.where(data["NEW_DOC_IND_KURT"]>0, data["DAYS_BIRTH"], data["NEW_DOC_IND_KURT"] )) + ((((data["DAYS_ID_PUBLISH"]) + (data["REG_CITY_NOT_WORK_CITY"]))/2.0)))))) * 2.0)) 
    v["i6"] = 0.099953*np.tanh(((np.where(np.where(data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]>0, (3.0), np.minimum(((((data["nejumi"]) - (data["NAME_EDUCATION_TYPE_Higher_education"])))), ((data["nejumi"]))) )<0, ((data["CC_CNT_DRAWINGS_POS_CURRENT_MAX"]) + (data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"])), (3.97629356384277344) )) * 2.0)) 
    v["i8"] = 0.099978*np.tanh(((np.where(data["nejumi"] < -99998, ((np.where(data["CODE_GENDER"]<0, 3.0, data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"] )) - (-1.0)), ((((np.tanh((data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]))) - (data["nejumi"]))) * 2.0) )) * 2.0)) 
    v["i13"] = 0.099900*np.tanh(((((((((np.where(data["nejumi"] < -99998, data["NEW_DOC_IND_KURT"], (((-1.0*((((1.570796) + (((data["nejumi"]) * 2.0))))))) * 2.0) )) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i17"] = 0.099948*np.tanh(((((np.maximum(((data["nejumi"])), ((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))) + ((((data["NEW_CREDIT_TO_INCOME_RATIO"]) + (data["NAME_INCOME_TYPE_Working"]))/2.0)))) + (np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]<0, np.minimum(((data["ORGANIZATION_TYPE_Self_employed"])), ((data["NEW_CREDIT_TO_INCOME_RATIO"]))), 1.570796 )))) 
    v["i25"] = 0.099954*np.tanh(((((((-1.0*((0.318310)))) < (data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]))*1.)) + (np.minimum((((-1.0*((((((data["nejumi"]) * 2.0)) * 2.0)))))), (((((data["ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN"]) > ((-1.0*((0.318310)))))*1.))))))) 
    v["i26"] = 0.099800*np.tanh((((np.maximum(((data["OCCUPATION_TYPE_Low_skill_Laborers"])), ((np.maximum(((data["ORGANIZATION_TYPE_Trade__type_3"])), ((data["nejumi"]))))))) + (np.minimum(((((((1.570796) + (((((data["NEW_CREDIT_TO_INCOME_RATIO"]) * 2.0)) * 2.0)))) * 2.0))), ((data["ORGANIZATION_TYPE_Trade__type_3"])))))/2.0)) 
    v["i35"] = 0.099982*np.tanh(np.where(data["nejumi"] < -99998, ((data["NAME_INCOME_TYPE_Maternity_leave"]) - (0.636620)), ((-2.0) - ((((8.16910552978515625)) * (data["nejumi"])))) )) 
    v["i47"] = 0.099954*np.tanh((((((((-1.0*((np.where(data["nejumi"] < -99998, (-1.0*((((0.318310) / 2.0)))), data["nejumi"] ))))) - (0.318310))) - ((((data["nejumi"]) > (data["ORGANIZATION_TYPE_Realtor"]))*1.)))) * 2.0)) 
    v["i50"] = 0.099990*np.tanh((-1.0*((np.where(data["nejumi"] < -99998, 0.636620, ((((((((((-1.0*((data["nejumi"])))) > (np.tanh((np.tanh((0.318310))))))*1.)) + (data["nejumi"]))) * 2.0)) * 2.0) ))))) 
    v["i51"] = 0.099807*np.tanh(np.where(data["ACTIVE_DAYS_CREDIT_VAR"] < -99998, ((((((0.636620) < (data["nejumi"]))*1.)) > (np.tanh((data["nejumi"]))))*1.), ((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"]) - ((-1.0*((data["nejumi"]))))))) )) 
    v["i57"] = 0.099809*np.tanh(((((((data["ORGANIZATION_TYPE_Transport__type_3"]) - (np.where(((data["nejumi"]) + (data["ORGANIZATION_TYPE_Legal_Services"])) < -99998, data["ORGANIZATION_TYPE_Legal_Services"], ((data["nejumi"]) + (0.636620)) )))) * 2.0)) * 2.0)) 
    v["i59"] = 0.099879*np.tanh(np.where(data["nejumi"] < -99998, data["OCCUPATION_TYPE_Low_skill_Laborers"], (((((((((((-1.0) > (data["nejumi"]))*1.)) - ((((np.tanh((0.318310))) < (data["nejumi"]))*1.)))) * 2.0)) * 2.0)) * 2.0) )) 
    v["i60"] = 0.096971*np.tanh(np.where(data["nejumi"]>0, ((data["AMT_INCOME_TOTAL"]) + ((-1.0*((((data["nejumi"]) * (((((data["FLAG_EMP_PHONE"]) / 2.0)) - (((data["nejumi"]) + (-2.0))))))))))), 0.318310 )) 
    v["i61"] = 0.099985*np.tanh(((np.where(data["nejumi"] < -99998, (8.0), data["nejumi"] )) * (np.minimum(((np.where(data["nejumi"] < -99998, (8.0), data["nejumi"] ))), ((np.where(data["nejumi"] < -99998, data["NAME_INCOME_TYPE_Unemployed"], -1.0 ))))))) 
    v["i65"] = 0.099989*np.tanh(np.where(data["nejumi"] < -99998, (((np.minimum(((data["DAYS_ID_PUBLISH"])), ((np.tanh((np.where(data["DAYS_ID_PUBLISH"]<0, data["ORGANIZATION_TYPE_Legal_Services"], -1.0 ))))))) + (data["DAYS_ID_PUBLISH"]))/2.0), ((data["ORGANIZATION_TYPE_Legal_Services"]) - (data["nejumi"])) )) 
    v["i74"] = 0.097890*np.tanh((((((((data["nejumi"]) > (3.141593))*1.)) + (np.where(data["nejumi"]>0, (((-1.0*((3.141593)))) * (data["FLAG_CONT_MOBILE"])), ((data["NAME_HOUSING_TYPE_With_parents"]) * (data["FLAG_CONT_MOBILE"])) )))) * 2.0)) 
    v["i76"] = 0.097098*np.tanh((-1.0*((((np.where(data["nejumi"] < -99998, np.tanh((np.tanh((np.tanh((data["nejumi"])))))), data["nejumi"] )) - (np.tanh((np.tanh((((((data["nejumi"]) * 2.0)) * 2.0))))))))))) 
    v["i81"] = 0.098999*np.tanh((((((-1.0*((((((((data["nejumi"]) > (((((((data["NAME_EDUCATION_TYPE_Academic_degree"]) + ((((data["nejumi"]) < (data["REG_CITY_NOT_WORK_CITY"]))*1.)))/2.0)) + (data["OCCUPATION_TYPE_Waiters_barmen_staff"]))/2.0)))*1.)) < (data["nejumi"]))*1.))))) * 2.0)) * 2.0)) 
    v["i82"] = 0.099260*np.tanh(np.where(data["nejumi"] < -99998, ((data["nejumi"]) - (data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"])), np.where(data["nejumi"]>0, data["CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN"], ((((1.74202001094818115)) < (((((data["nejumi"]) * (data["nejumi"]))) / 2.0)))*1.) ) )) 
    v["i84"] = 0.099990*np.tanh(((((data["nejumi"]) * 2.0)) * (((((data["nejumi"]) * 2.0)) - (np.minimum(((np.where(data["nejumi"] < -99998, ((data["nejumi"]) * 2.0), (-1.0*((data["nejumi"]))) ))), ((data["NEW_CREDIT_TO_INCOME_RATIO"])))))))) 
    v["i85"] = 0.099710*np.tanh((((((((data["nejumi"]) * (data["nejumi"]))) * (((((np.tanh((data["nejumi"]))) * (data["nejumi"]))) * 2.0)))) + (((data["nejumi"]) * (data["nejumi"]))))/2.0)) 
    v["i99"] = 0.100000*np.tanh(((((np.tanh((np.minimum(((np.tanh((np.tanh((np.tanh((((((((data["AMT_ANNUITY"]) * 2.0)) * 2.0)) * 2.0))))))))), ((0.636620)))))) - (data["nejumi"]))) * ((4.0)))) 
    v["i100"] = 0.099095*np.tanh(((np.tanh((data["AMT_CREDIT"]))) * (np.where(np.where(data["NAME_CONTRACT_TYPE_Revolving_loans"]<0, data["nejumi"], 0.636620 ) < -99998, (-1.0*((data["AMT_CREDIT"]))), data["nejumi"] )))) 
    v["i102"] = 0.098001*np.tanh(((np.maximum((((-1.0*((data["DAYS_BIRTH"]))))), ((np.where(data["nejumi"]<0, data["DAYS_BIRTH"], data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"] ))))) * ((-1.0*((np.maximum((((((data["ORGANIZATION_TYPE_Transport__type_1"]) + (data["DAYS_BIRTH"]))/2.0))), ((data["NAME_HOUSING_TYPE_Office_apartment"]))))))))) 
    v["i103"] = 0.094951*np.tanh(np.where(((data["ORGANIZATION_TYPE_Mobile"]) * (data["nejumi"]))>0, ((data["WALLSMATERIAL_MODE_Wooden"]) * (data["NAME_EDUCATION_TYPE_Higher_education"])), (((((data["NAME_EDUCATION_TYPE_Higher_education"]) / 2.0)) + ((((data["NAME_INCOME_TYPE_Maternity_leave"]) + ((-1.0*((data["nejumi"])))))/2.0)))/2.0) )) 
    v["i121"] = 0.099190*np.tanh(np.where(np.minimum(((data["nejumi"])), (((-1.0*((data["NAME_EDUCATION_TYPE_Higher_education"]))))))<0, ((((data["NAME_EDUCATION_TYPE_Academic_degree"]) - (data["NAME_EDUCATION_TYPE_Higher_education"]))) * ((((data["NAME_CONTRACT_TYPE_Revolving_loans"]) + (data["NAME_INCOME_TYPE_Student"]))/2.0))), data["OCCUPATION_TYPE_Core_staff"] )) 
    v["i128"] = 0.093700*np.tanh(np.where(((data["nejumi"]) + (1.0))>0, ((((np.minimum(((0.318310)), ((((1.0) + (data["nejumi"])))))) + (data["nejumi"]))) * 2.0), data["ORGANIZATION_TYPE_Legal_Services"] )) 
    v["i176"] = 0.098959*np.tanh(np.where((((data["nejumi"]) + (data["NAME_INCOME_TYPE_Pensioner"]))/2.0)>0, data["NAME_INCOME_TYPE_Pensioner"], np.where(data["nejumi"] < -99998, np.minimum(((data["NAME_INCOME_TYPE_Maternity_leave"])), (((-1.0*(((((data["NAME_INCOME_TYPE_Pensioner"]) + (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]))/2.0))))))), data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] ) )) 
    v["i177"] = 0.099800*np.tanh((((((np.maximum(((((((data["nejumi"]) * 2.0)) * (data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"])))), ((np.maximum(((((data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]) * (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"])))), ((0.636620))))))) * (-2.0))) > (data["nejumi"]))*1.)) 
    v["i179"] = 0.099995*np.tanh(np.where(data["nejumi"]<0, data["NAME_INCOME_TYPE_Maternity_leave"], (((((data["nejumi"]) > (data["NAME_EDUCATION_TYPE_Higher_education"]))*1.)) + (np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]<0, 0.318310, (-1.0*((data["nejumi"]))) ))) )) 
    v["i181"] = 0.080049*np.tanh(np.minimum(((np.where(((((-1.0) / 2.0)) - (data["nejumi"]))>0, (((data["nejumi"]) > (data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.), ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) / 2.0) ))), (((((data["CLOSED_AMT_CREDIT_SUM_LIMIT_SUM"]) < (data["nejumi"]))*1.))))) 
    v["i184"] = 0.099950*np.tanh((-1.0*((((((np.where(data["ORGANIZATION_TYPE_XNA"]>0, (((data["nejumi"]) > (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.), ((((-1.0*(((((data["nejumi"]) > (1.570796))*1.))))) < (data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]))*1.) )) * 2.0)) * 2.0))))) 
    v["i185"] = 0.099008*np.tanh(np.where(data["nejumi"]>0, (-1.0*((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))), (((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]) > (np.maximum((((((data["nejumi"]) + ((((data["nejumi"]) + ((5.0)))/2.0)))/2.0))), (((-1.0*((0.318310))))))))*1.) )) 
    v["i189"] = 0.099484*np.tanh(np.maximum(((((np.where(data["APPROVED_AMT_DOWN_PAYMENT_MAX"]>0, ((data["nejumi"]) * 2.0), data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"] )) + (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"])))), ((((np.where(data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]<0, data["NEW_CREDIT_TO_INCOME_RATIO"], 0.318310 )) * (0.636620)))))) 
    v["i194"] = 0.098200*np.tanh((((((np.tanh((data["nejumi"]))) > ((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MAX"]) < (((((((data["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"]) > (1.0))*1.)) + ((((1.570796) < (data["nejumi"]))*1.)))/2.0)))*1.)))*1.)) * ((4.11938047409057617)))) 
    v["i196"] = 0.099004*np.tanh(((((((np.tanh((((0.636620) - (data["REFUSED_RATE_DOWN_PAYMENT_MAX"]))))) < (np.where(data["REG_REGION_NOT_LIVE_REGION"]>0, ((0.636620) - (data["nejumi"])), data["nejumi"] )))*1.)) < (data["REFUSED_AMT_DOWN_PAYMENT_MEAN"]))*1.)) 
    v["i197"] = 0.094900*np.tanh((-1.0*((np.where((((1.0) < (data["nejumi"]))*1.)>0, (((data["OCCUPATION_TYPE_Private_service_staff"]) + ((((((1.570796) + (data["REG_REGION_NOT_LIVE_REGION"]))) < (data["nejumi"]))*1.)))/2.0), data["REG_REGION_NOT_LIVE_REGION"] ))))) 
    v["i257"] = 0.098100*np.tanh(np.minimum((((((((data["nejumi"]) > ((((-1.0*((data["OCCUPATION_TYPE_Managers"])))) * 2.0)))*1.)) * 2.0))), ((np.where(data["nejumi"]<0, data["OCCUPATION_TYPE_Managers"], (((((-1.0*((data["OCCUPATION_TYPE_Managers"])))) * 2.0)) * 2.0) ))))) 
    v["i258"] = 0.097465*np.tanh((((((-1.0*(((((data["nejumi"]) > (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.))))) * (data["nejumi"]))) * (np.minimum((((((data["nejumi"]) > (1.0))*1.))), ((data["nejumi"])))))) 
    v["i261"] = 0.097180*np.tanh(((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * ((-1.0*((((((((data["nejumi"]) < ((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + (data["NEW_CAR_TO_EMPLOY_RATIO"]))/2.0)))*1.)) + ((((data["nejumi"]) < (np.minimum(((-1.0)), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))))*1.)))/2.0))))))) 
    v["i268"] = 0.099500*np.tanh((((((data["NAME_INCOME_TYPE_Maternity_leave"]) > (np.minimum(((((3.141593) - (data["nejumi"])))), ((((data["nejumi"]) + (1.570796)))))))*1.)) + ((-1.0*(((((data["ACTIVE_DAYS_CREDIT_VAR"]) > (3.0))*1.))))))) 
    v["i274"] = 0.099951*np.tanh(np.where(((data["NEW_CREDIT_TO_INCOME_RATIO"]) - (1.570796))<0, np.where(np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, data["CLOSED_DAYS_CREDIT_VAR"], data["nejumi"] )>0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data["NAME_INCOME_TYPE_Maternity_leave"] ), ((data["CLOSED_DAYS_CREDIT_VAR"]) - (data["nejumi"])) )) 
    v["i276"] = 0.078515*np.tanh((-1.0*(((((((((3.141593) < (((3.141593) * (data["AMT_ANNUITY"]))))*1.)) * ((((data["nejumi"]) + (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (data["AMT_CREDIT"]))))/2.0)))) * (data["AMT_CREDIT"])))))) 
    v["i279"] = 0.098040*np.tanh(((((((((data["nejumi"]) * ((((1.570796) < (data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]))*1.)))) - ((((data["nejumi"]) > (1.570796))*1.)))) - ((((data["nejumi"]) > (1.570796))*1.)))) * 2.0)) 
    v["i280"] = 0.092470*np.tanh((-1.0*(((((np.where(data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"]<0, (-1.0*((np.where(data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"]<0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], np.where(data["nejumi"]<0, data["CLOSED_DAYS_CREDIT_ENDDATE_MEAN"], data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] ) )))), data["CLOSED_DAYS_CREDIT_UPDATE_MEAN"] )) < (-1.0))*1.))))) 
    v["i282"] = 0.042521*np.tanh(np.tanh(((((-1.0*((np.tanh((((((0.636620) - ((-1.0*((data["nejumi"])))))) * (np.where(data["AMT_CREDIT"]>0, data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"], data["NEW_CAR_TO_EMPLOY_RATIO"] ))))))))) / 2.0)))) 
    v["i293"] = 0.099000*np.tanh(((3.0) * ((((data["nejumi"]) < (np.where(data["nejumi"] < -99998, data["ACTIVE_AMT_ANNUITY_MAX"], np.where(data["nejumi"]>0, data["BURO_AMT_ANNUITY_MEAN"], np.where(data["ACTIVE_AMT_ANNUITY_MAX"]>0, data["nejumi"], -2.0 ) ) )))*1.)))) 
    v["i294"] = 0.010870*np.tanh(np.tanh(((((data["AMT_ANNUITY"]) < (np.minimum(((np.where((((0.318310) + (data["AMT_ANNUITY"]))/2.0)<0, data["BURO_AMT_ANNUITY_MAX"], data["nejumi"] ))), ((0.318310)))))*1.)))) 
    v["i295"] = 0.005211*np.tanh(((np.where((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (3.141593))*1.)>0, ((np.where(((data["ORGANIZATION_TYPE_Trade__type_4"]) * (data["NAME_INCOME_TYPE_Student"]))>0, ((data["NAME_INCOME_TYPE_Maternity_leave"]) * 2.0), data["nejumi"] )) * 2.0), data["nejumi"] )) * 2.0)) 
    v["i297"] = 0.099999*np.tanh((((np.where(np.where(data["nejumi"] < -99998, data["PREV_AMT_DOWN_PAYMENT_MAX"], data["nejumi"] ) < -99998, -2.0, 0.318310 )) + (((np.where(data["nejumi"]<0, -1.0, data["PREV_AMT_DOWN_PAYMENT_MAX"] )) / 2.0)))/2.0)) 
    v["i307"] = 0.097210*np.tanh(np.tanh((((((((1.0) + (data["CLOSED_DAYS_CREDIT_VAR"]))) + (data["nejumi"]))) * ((((np.tanh((np.tanh((-1.0))))) < ((((data["nejumi"]) + (data["AMT_REQ_CREDIT_BUREAU_DAY"]))/2.0)))*1.)))))) 
    v["i308"] = 0.034689*np.tanh(np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), (((((0.636620) < (np.where(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]<0, data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"], data["nejumi"] )<0, np.where(data["nejumi"] < -99998, data["BURO_DAYS_CREDIT_ENDDATE_MEAN"], data["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] ), data["nejumi"] )))*1.))))) 
    v["i311"] = 0.091399*np.tanh(((np.where((((data["nejumi"]) + (1.0))/2.0)>0, ((data["nejumi"]) * 2.0), np.minimum((((-1.0*((data["nejumi"]))))), (((((data["nejumi"]) > (data["nejumi"]))*1.)))) )) / 2.0)) 
    v["i312"] = 0.080001*np.tanh(np.where(data["nejumi"]>0, np.minimum(((data["nejumi"])), ((data["AMT_REQ_CREDIT_BUREAU_YEAR"]))), ((0.0) - ((((0.318310) < (np.minimum(((data["AMT_REQ_CREDIT_BUREAU_QRT"])), ((data["nejumi"])))))*1.))) )) 
    v["i313"] = 0.095050*np.tanh(np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), ((np.where(data["nejumi"]<0, (((data["AMT_REQ_CREDIT_BUREAU_MON"]) > (np.where(data["nejumi"] < -99998, data["AMT_REQ_CREDIT_BUREAU_MON"], ((data["nejumi"]) + ((0.97451114654541016))) )))*1.), data["AMT_REQ_CREDIT_BUREAU_MON"] ))))) 
    v["i315"] = 0.095929*np.tanh(((((((((data["AMT_REQ_CREDIT_BUREAU_HOUR"]) < (data["AMT_REQ_CREDIT_BUREAU_MON"]))*1.)) < (((((((data["nejumi"]) < (data["AMT_REQ_CREDIT_BUREAU_WEEK"]))*1.)) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))*1.)) * (((((data["nejumi"]) * 2.0)) + (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))))) 
    v["i319"] = 0.099702*np.tanh(((data["nejumi"]) * ((((np.maximum(((0.318310)), ((data["nejumi"])))) < (np.minimum(((data["nejumi"])), ((((((((data["nejumi"]) + (data["NAME_INCOME_TYPE_Student"]))/2.0)) > ((-1.0*((0.318310)))))*1.))))))*1.)))) 
    v["i338"] = 0.097991*np.tanh(np.where(data["ORGANIZATION_TYPE_Business_Entity_Type_1"]<0, (((((2.0) < (((np.tanh((data["nejumi"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))))*1.)) * 2.0), ((((data["nejumi"]) - (data["NEW_ANNUITY_TO_INCOME_RATIO"]))) - (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )) 
    v["i341"] = 0.098498*np.tanh(np.where(np.maximum(((((data["nejumi"]) + (((((data["nejumi"]) / 2.0)) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])))))), ((data["nejumi"])))<0, data["ORGANIZATION_TYPE_Telecom"], ((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["NEW_ANNUITY_TO_INCOME_RATIO"])) )) 
    v["i342"] = 0.097800*np.tanh(((((np.where(data["nejumi"]>0, ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["nejumi"], -1.0 )) / 2.0), np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]>0, data["nejumi"], data["AMT_INCOME_TOTAL"] ) )) * (data["NAME_EDUCATION_TYPE_Secondary___secondary_special"]))) / 2.0)) 
    v["i343"] = 0.098500*np.tanh(((((np.tanh((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]>0, (-1.0*((data["nejumi"]))), ((((-1.0*(((0.35045036673545837))))) < (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))*1.) )))) / 2.0)) - ((((3.0) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.)))) 
    v["i345"] = 0.099994*np.tanh((((((-1.0*((data["nejumi"])))) * 2.0)) - (np.where(data["nejumi"]<0, (((((data["nejumi"]) * 2.0)) < (data["NEW_CREDIT_TO_INCOME_RATIO"]))*1.), (((-1.0) + (((data["nejumi"]) * 2.0)))/2.0) )))) 
    v["i376"] = 0.098411*np.tanh(((np.minimum(((data["WALLSMATERIAL_MODE_Stone__brick"])), ((np.where(data["nejumi"]<0, np.minimum(((data["AMT_ANNUITY"])), ((np.where(data["WALLSMATERIAL_MODE_Stone__brick"]<0, data["NEW_CREDIT_TO_INCOME_RATIO"], data["ORGANIZATION_TYPE_School"] )))), data["ORGANIZATION_TYPE_School"] ))))) - (data["nejumi"]))) 
    v["i377"] = 0.100000*np.tanh((-1.0*((np.where(data["nejumi"]<0, ((np.where((((data["nejumi"]) < ((-1.0*((data["DAYS_BIRTH"])))))*1.)>0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN"], ((data["REGION_RATING_CLIENT"]) * 2.0) )) - (data["nejumi"])), data["nejumi"] ))))) 
    v["i378"] = 0.074051*np.tanh((-1.0*(((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > ((-1.0*((data["DAYS_BIRTH"])))))*1.)) * ((((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) > (np.tanh((np.tanh((np.tanh((data["nejumi"]))))))))*1.)) * (data["DAYS_BIRTH"])))))))) 
    v["i379"] = 0.095001*np.tanh((((data["nejumi"]) > (((np.maximum(((np.maximum(((np.maximum(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) * (data["AMT_REQ_CREDIT_BUREAU_YEAR"])))), ((data["nejumi"]))))), ((data["AMT_REQ_CREDIT_BUREAU_QRT"]))))), ((np.tanh((3.141593)))))) + (data["BURO_DAYS_CREDIT_ENDDATE_MEAN"]))))*1.)) 
    v["i380"] = 0.100000*np.tanh(np.where(data["nejumi"] < -99998, (-1.0*((((1.0) + (data["nejumi"]))))), ((((0.636620) + (((((((0.636620) + (data["nejumi"]))) * 2.0)) * 2.0)))) * 2.0) )) 
    v["i381"] = 0.099600*np.tanh(((((((((data["REGION_RATING_CLIENT_W_CITY"]) * (0.318310))) - ((((data["nejumi"]) < (((np.tanh((data["nejumi"]))) + (-2.0))))*1.)))) * (data["nejumi"]))) * 2.0)) 
    v["i382"] = 0.099995*np.tanh(np.where(((((0.61522376537322998)) + (data["nejumi"]))/2.0)<0, (((((0.318310) - (data["nejumi"]))) > ((-1.0*((-2.0)))))*1.), (-1.0*(((((data["nejumi"]) + (data["nejumi"]))/2.0)))) )) 
    v["i383"] = 0.099400*np.tanh(((np.minimum(((((((2.0) + (data["REGION_RATING_CLIENT"]))) + (data["DAYS_BIRTH"])))), (((((-1.0*((data["nejumi"])))) + (-2.0)))))) * (data["DAYS_BIRTH"]))) 
    v["i384"] = 0.099699*np.tanh(((((((((data["DAYS_REGISTRATION"]) > (0.0))*1.)) - (data["CODE_GENDER"]))) < (np.minimum(((data["REGION_POPULATION_RELATIVE"])), ((np.minimum(((data["nejumi"])), ((data["WALLSMATERIAL_MODE_Panel"]))))))))*1.)) 
    v["i385"] = 0.089970*np.tanh((((((((np.maximum((((((np.minimum(((data["nejumi"])), ((data["nejumi"])))) + (2.0))/2.0))), ((((np.minimum(((1.570796)), ((data["nejumi"])))) / 2.0))))) < (data["NEW_CAR_TO_BIRTH_RATIO"]))*1.)) * 2.0)) * 2.0)) 
    v["i386"] = 0.099006*np.tanh((((2.52555441856384277)) * (np.minimum(((np.maximum(((((((1.0) - (data["nejumi"]))) * 2.0))), ((data["REFUSED_AMT_DOWN_PAYMENT_MAX"]))))), (((((0.636620) < (data["nejumi"]))*1.))))))) 
    v["i388"] = 0.099990*np.tanh(np.where(data["NAME_FAMILY_STATUS_Single___not_married"]>0, (-1.0*((((data["DAYS_BIRTH"]) + (np.minimum(((data["NAME_INCOME_TYPE_Student"])), ((data["REG_CITY_NOT_WORK_CITY"])))))))), np.maximum(((data["NAME_INCOME_TYPE_Student"])), (((-1.0*((((data["nejumi"]) + ((1.65381467342376709))))))))) )) 
    v["i389"] = 0.082100*np.tanh(((data["NAME_EDUCATION_TYPE_Higher_education"]) * (((data["nejumi"]) * ((((((np.maximum(((((data["nejumi"]) * (data["HOUR_APPR_PROCESS_START"])))), ((data["HOUR_APPR_PROCESS_START"])))) > (1.570796))*1.)) + (((-1.0) / 2.0)))))))) 
    v["i390"] = 0.072495*np.tanh(((data["AMT_REQ_CREDIT_BUREAU_DAY"]) * ((((data["nejumi"]) > (((np.tanh(((((((data["nejumi"]) > (data["AMT_REQ_CREDIT_BUREAU_DAY"]))*1.)) + (np.minimum(((data["nejumi"])), ((0.318310)))))))) + (data["HOUR_APPR_PROCESS_START"]))))*1.)))) 
    v["i393"] = 0.050400*np.tanh(np.where(data["nejumi"] < -99998, data["nejumi"], np.where((((data["nejumi"]) + (data["OCCUPATION_TYPE_High_skill_tech_staff"]))/2.0)>0, (-1.0*((data["NAME_TYPE_SUITE_Group_of_people"]))), ((((((data["nejumi"]) + (2.0))/2.0)) < (data["OCCUPATION_TYPE_Core_staff"]))*1.) ) )) 
    v["i394"] = 0.096996*np.tanh((((((data["OCCUPATION_TYPE_High_skill_tech_staff"]) > (((data["nejumi"]) * (data["OCCUPATION_TYPE_High_skill_tech_staff"]))))*1.)) * (np.where(((data["nejumi"]) * (data["REG_CITY_NOT_WORK_CITY"]))>0, data["OCCUPATION_TYPE_High_skill_tech_staff"], data["WALLSMATERIAL_MODE_Panel"] )))) 
    v["i395"] = 0.075010*np.tanh(((((((np.maximum((((((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]) > ((((np.minimum(((data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"])), ((data["NAME_INCOME_TYPE_Student"])))) + (((1.570796) - (data["nejumi"]))))/2.0)))*1.))), ((data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])))) * 2.0)) * 2.0)) * 2.0)) 
    v["i398"] = 0.099104*np.tanh(((((np.where(data["nejumi"] < -99998, data["nejumi"], data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] )) - (np.where(np.minimum(((data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"])), (((((data["nejumi"]) > (data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]))*1.))))>0, data["ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], data["ACTIVE_CREDIT_DAY_OVERDUE_MEAN"] )))) * 2.0)) 
    v["i400"] = 0.100000*np.tanh(np.where(data["ACTIVE_MONTHS_BALANCE_SIZE_SUM"]>0, data["nejumi"], np.where(data["ACTIVE_DAYS_CREDIT_UPDATE_MEAN"]<0, np.where((((data["nejumi"]) > (1.570796))*1.)>0, data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"], (((data["ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM"]) > (0.636620))*1.) ), data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] ) )) 
    v["i401"] = 0.090000*np.tanh(np.where(np.where(data["ACTIVE_CNT_CREDIT_PROLONG_SUM"]>0, (-1.0*((data["nejumi"]))), data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"] )>0, ((data["ACTIVE_DAYS_CREDIT_MEAN"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"])), (((1.570796) < (((data["nejumi"]) + (data["ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN"]))))*1.) )) 
    v["i404"] = 0.099100*np.tanh(np.where(data["nejumi"] < -99998, data["nejumi"], ((data["CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN"]) * ((-1.0*(((((((0.318310) * ((((((data["CLOSED_DAYS_CREDIT_VAR"]) / 2.0)) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))) < (data["ACTIVE_DAYS_CREDIT_VAR"]))*1.)))))) )) 
    v["i405"] = 0.098018*np.tanh((-1.0*((np.where(np.where(np.where(data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"] < -99998, ((data["ACTIVE_DAYS_CREDIT_VAR"]) + (3.0)), data["CLOSED_AMT_CREDIT_SUM_DEBT_MEAN"] )>0, data["nejumi"], data["NAME_INCOME_TYPE_Student"] )>0, (4.06813240051269531), data["NAME_INCOME_TYPE_Student"] ))))) 
    v["i407"] = 0.099599*np.tanh(np.where(data["ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN"]>0, (((data["DAYS_BIRTH"]) + (((data["nejumi"]) * 2.0)))/2.0), np.maximum((((((data["BURO_AMT_CREDIT_MAX_OVERDUE_MEAN"]) > ((0.04560829326510429)))*1.))), (((((-1.0*(((0.17060284316539764))))) * (data["DAYS_BIRTH"]))))) )) 
    v["i408"] = 0.100000*np.tanh(np.where(data["nejumi"] < -99998, data["nejumi"], (((0.0) > (((np.where(data["ACTIVE_MONTHS_BALANCE_MIN_MIN"]<0, data["nejumi"], np.where(data["nejumi"]<0, data["ACTIVE_MONTHS_BALANCE_MAX_MAX"], data["YEARS_BUILD_AVG"] ) )) + ((2.71755051612854004)))))*1.) )) 
    v["i409"] = 0.050000*np.tanh(np.where(data["OCCUPATION_TYPE_Secretaries"]>0, 3.0, np.where(data["nejumi"]>0, data["POS_SK_DPD_MEAN"], (((np.where(data["POS_SK_DPD_MEAN"]>0, 3.0, (((data["OCCUPATION_TYPE_Secretaries"]) + (data["NAME_TYPE_SUITE_Group_of_people"]))/2.0) )) < (data["POS_SK_DPD_MEAN"]))*1.) ) )) 
    v["i411"] = 0.088340*np.tanh(((((((((np.where(data["nejumi"]<0, 1.570796, 0.636620 )) + (1.0))/2.0)) < (data["nejumi"]))*1.)) * (((((data["nejumi"]) - (1.0))) * (3.0))))) 
    v["i412"] = 0.098500*np.tanh((-1.0*((((((((((data["nejumi"]) + (data["nejumi"]))/2.0)) + ((((((data["nejumi"]) + (((data["nejumi"]) + (data["nejumi"]))))) > (1.570796))*1.)))) > (1.570796))*1.))))) 
    v["i414"] = 0.099499*np.tanh(np.where(data["BURO_DAYS_CREDIT_ENDDATE_MEAN"] < -99998, data["AMT_ANNUITY"], np.where((((np.tanh((0.636620))) + (np.where(data["nejumi"]<0, np.tanh((data["BURO_DAYS_CREDIT_ENDDATE_MEAN"])), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )))/2.0)<0, 3.0, data["NAME_INCOME_TYPE_Maternity_leave"] ) )) 
    v["i417"] = 0.099799*np.tanh(np.where((((data["nejumi"]) < (data["CC_AMT_PAYMENT_CURRENT_MEAN"]))*1.)>0, ((data["nejumi"]) - (data["AMT_ANNUITY"])), (((data["CC_AMT_PAYMENT_CURRENT_MAX"]) > (np.where(data["CC_AMT_PAYMENT_CURRENT_MEAN"]<0, ((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0), 1.0 )))*1.) )) 
    v["i418"] = 0.099600*np.tanh((((np.where(((((((-2.0) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) / 2.0)) - (((data["nejumi"]) * 2.0)))<0, np.tanh((data["CC_CNT_DRAWINGS_CURRENT_MEAN"])), data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) > (np.tanh((0.636620))))*1.)) 
    v["i419"] = 0.094997*np.tanh((-1.0*((((((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) + ((((0.318310) < ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) + (data["nejumi"]))/2.0)))*1.)))/2.0)) + (((((((0.318310) + (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))/2.0)) < (data["ORGANIZATION_TYPE_Cleaning"]))*1.)))/2.0))))) 
    v["i420"] = 0.080000*np.tanh(((data["nejumi"]) * (((data["DAYS_BIRTH"]) * ((-1.0*(((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) > ((((np.tanh((data["NEW_CREDIT_TO_ANNUITY_RATIO"]))) > (((1.0) / 2.0)))*1.)))*1.))))))))) 
    v["i421"] = 0.099805*np.tanh((((np.where(data["DAYS_BIRTH"]<0, ((np.where(data["NEW_ANNUITY_TO_INCOME_RATIO"]<0, 2.0, data["nejumi"] )) * (data["NEW_ANNUITY_TO_INCOME_RATIO"])), np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, 2.0, data["nejumi"] ) )) < (-2.0))*1.)) 
    v["i422"] = 0.099900*np.tanh(np.where(data["nejumi"] < -99998, data["nejumi"], (((np.where(data["NEW_CREDIT_TO_INCOME_RATIO"]<0, ((data["nejumi"]) * (data["AMT_CREDIT"])), ((data["AMT_CREDIT"]) * (1.570796)) )) < ((-1.0*((0.636620)))))*1.) )) 
    v["i423"] = 0.098598*np.tanh(((np.maximum(((data["nejumi"])), ((((data["AMT_ANNUITY"]) / 2.0))))) * (((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) - ((((np.where(data["nejumi"]<0, data["AMT_ANNUITY"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) > (((data["AMT_ANNUITY"]) / 2.0)))*1.)))))) 
    v["i424"] = 0.099600*np.tanh((-1.0*((((((data["nejumi"]) * 2.0)) * (np.maximum(((data["NAME_TYPE_SUITE_Group_of_people"])), ((((((data["ACTIVE_MONTHS_BALANCE_SIZE_MEAN"]) + (((data["nejumi"]) / 2.0)))) + (((((data["NEW_CREDIT_TO_ANNUITY_RATIO"]) * 2.0)) * 2.0)))))))))))) 
    v["i425"] = 0.000051*np.tanh((((((((data["ORGANIZATION_TYPE_Trade__type_3"]) + (((data["nejumi"]) * (data["NEW_CREDIT_TO_ANNUITY_RATIO"]))))/2.0)) * (((((data["ORGANIZATION_TYPE_Trade__type_3"]) * ((((-1.0*((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) - (data["nejumi"]))))) * 2.0)))) / 2.0)) 
    v["i429"] = 0.094973*np.tanh((((((((3.0) < (data["NEW_ANNUITY_TO_INCOME_RATIO"]))*1.)) * (((((-1.0) + (((((data["nejumi"]) + (3.0))) * (((data["nejumi"]) * 2.0)))))) * 2.0)))) * 2.0)) 
    v["i430"] = 0.100000*np.tanh(np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]<0, 0.0, ((np.where(data["NEW_CREDIT_TO_ANNUITY_RATIO"]<0, data["NEW_ANNUITY_TO_INCOME_RATIO"], np.where(data["nejumi"]<0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], -2.0 ) )) + (((data["nejumi"]) - (data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) )) 
    v["i431"] = 0.099000*np.tanh(((((((-2.0) + (data["NAME_INCOME_TYPE_Student"]))) + (((((np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["nejumi"])))) / 2.0)) * (data["nejumi"]))))) * (np.maximum(((data["BURO_CREDIT_TYPE_Mortgage_MEAN"])), ((data["NAME_TYPE_SUITE_Group_of_people"])))))) 
    v["i433"] = 0.068043*np.tanh((((((data["ORGANIZATION_TYPE_Trade__type_3"]) > ((((np.minimum(((data["REG_CITY_NOT_WORK_CITY"])), ((data["AMT_ANNUITY"])))) + ((((data["nejumi"]) < (1.570796))*1.)))/2.0)))*1.)) * ((-1.0*((data["REG_CITY_NOT_WORK_CITY"])))))) 
    v["i436"] = 0.055846*np.tanh((-1.0*((((((((data["AMT_ANNUITY"]) * (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))) + (data["AMT_ANNUITY"]))) * ((((data["CLOSED_MONTHS_BALANCE_MAX_MAX"]) > ((((((data["AMT_ANNUITY"]) * (data["CLOSED_MONTHS_BALANCE_MIN_MIN"]))) > (data["nejumi"]))*1.)))*1.))))))) 
    v["i437"] = 0.086880*np.tanh(((data["NEW_ANNUITY_TO_INCOME_RATIO"]) * (((np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"] < -99998, data["nejumi"], data["NEW_ANNUITY_TO_INCOME_RATIO"] )) * (np.where(data["CLOSED_MONTHS_BALANCE_SIZE_MEAN"]>0, data["NEW_CREDIT_TO_ANNUITY_RATIO"], (((data["NEW_ANNUITY_TO_INCOME_RATIO"]) > (2.0))*1.) )))))) 
    v["i439"] = 0.086331*np.tanh((-1.0*(((((data["CLOSED_MONTHS_BALANCE_MIN_MIN"]) > (((2.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MAX_MAX"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], ((0.0) - (np.where(data["CLOSED_MONTHS_BALANCE_MIN_MIN"]>0, data["nejumi"], data["CLOSED_MONTHS_BALANCE_MAX_MAX"] ))) )))))*1.))))) 
    v["i440"] = 0.099890*np.tanh(np.where((((((np.tanh((0.318310))) / 2.0)) > (data["nejumi"]))*1.)>0, ((data["nejumi"]) * ((((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["nejumi"]))*1.))), (((-1.0*((data["nejumi"])))) / 2.0) )) 
    v["i456"] = 0.051000*np.tanh(np.minimum(((np.maximum(((data["BURO_CREDIT_ACTIVE_Closed_MEAN"])), ((0.0))))), ((((np.minimum(((((((((((-2.0) + (0.318310))/2.0)) / 2.0)) > (data["nejumi"]))*1.))), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) * (data["BURO_CREDIT_ACTIVE_Closed_MEAN"])))))) 
    v["i457"] = 0.099734*np.tanh((-1.0*(((((np.where(np.maximum(((data["NEW_ANNUITY_TO_INCOME_RATIO"])), ((data["ACTIVE_DAYS_CREDIT_VAR"])))>0, data["NAME_INCOME_TYPE_Student"], (((-1.0*((data["nejumi"])))) / 2.0) )) < (((-2.0) + (data["nejumi"]))))*1.))))) 
    v["i461"] = 0.004245*np.tanh((-1.0*((((np.minimum(((data["AMT_CREDIT"])), ((data["AMT_ANNUITY"])))) * ((((data["YEARS_BUILD_AVG"]) > (((((((-1.0) + ((((data["AMT_CREDIT"]) < (data["AMT_ANNUITY"]))*1.)))/2.0)) < (data["nejumi"]))*1.)))*1.))))))) 
    v["i463"] = 0.099000*np.tanh(np.minimum((((((np.maximum(((data["nejumi"])), ((data["NEW_CREDIT_TO_ANNUITY_RATIO"])))) < (np.maximum(((data["CC_AMT_PAYMENT_CURRENT_VAR"])), ((1.570796)))))*1.))), ((((data["nejumi"]) * ((((data["NEW_ANNUITY_TO_INCOME_RATIO"]) < (data["CC_AMT_PAYMENT_CURRENT_VAR"]))*1.))))))) 
    v["i510"] = 0.069406*np.tanh((((((((np.where(((np.where(data["nejumi"]>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], 0.636620 )) + (data["NEW_ANNUITY_TO_INCOME_RATIO"]))>0, data["NEW_ANNUITY_TO_INCOME_RATIO"], data["NEW_CREDIT_TO_ANNUITY_RATIO"] )) - (data["nejumi"]))) > (3.0))*1.)) * 2.0)) 
    v["i511"] = 0.093680*np.tanh((((((np.where(data["BURO_AMT_CREDIT_SUM_LIMIT_MEAN"] < -99998, data["nejumi"], np.where(data["nejumi"] < -99998, data["AMT_ANNUITY"], (((data["NAME_INCOME_TYPE_Maternity_leave"]) < (data["CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN"]))*1.) ) )) > ((((data["AMT_ANNUITY"]) > (data["nejumi"]))*1.)))*1.)) * 2.0))
    return v.add_prefix('gp3_')



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv.zip', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv.zip', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
#    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv.zip', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv.zip', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bb_agg.reset_index(inplace=True)
    bureau = pd.merge(bureau, bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau_agg.reset_index(inplace=True)
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    active_agg.reset_index(inplace=True)
    bureau_agg = pd.merge(bureau_agg, active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    closed_agg.reset_index(inplace=True)
    bureau_agg = pd.merge(bureau_agg, closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv.zip', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg.reset_index(inplace=True)
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    approved_agg.reset_index(inplace=True)
    prev_agg = pd.merge(prev_agg, approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    refused_agg.reset_index(inplace=True)
    prev_agg = pd.merge(prev_agg, refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv.zip', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv.zip', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv.zip', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

debug = None
num_rows = 10000 if debug else None
df = application_train_test(num_rows)
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = pd.merge(df, bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = pd.merge(df, prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows).reset_index()
    print("Pos-cash balance df shape:", pos.shape)
    df = pd.merge(df, pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows).reset_index()
    print("Installments payments df shape:", ins.shape)
    df = pd.merge(df, ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows).reset_index()
    print("Credit card balance df shape:", cc.shape)
    df = pd.merge(df, cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

for c in feats:
    print(c)
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
    df[c].fillna(-99999.,inplace=True)

train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print(train_df.shape)
print(test_df.shape)

train_df.columns = train_df.columns.str.replace('[^A-Za-z0-9_]', '_')
test_df.columns = test_df.columns.str.replace('[^A-Za-z0-9_]', '_')
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

train_df['nejumi'] = np.load('../feature_someone/train_nejumi.npy')
test_df['nejumi']  = np.load('../feature_someone/test_nejumi.npy')


train_gp = pd.concat([GP1(train_df.round(6)), GP2(train_df.round(6)), GP3(train_df.round(6))], axis=1)
test_gp  = pd.concat([GP1(test_df.round(6)),  GP2(test_df.round(6)),  GP3(test_df.round(6))], axis=1)


utils.to_pkl_gzip(train_gp, '../data/X_train_nejumi_gp.pkl')
utils.to_pkl_gzip(test_gp,  '../data/X_test_nejumi_gp.pkl')

#train_gp.to_feather('../data/X_train_pure')
#test_gp.to_feather()




#==============================================================================
utils.end(__file__)












