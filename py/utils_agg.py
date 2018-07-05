#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:27:07 2018

@author: kazuki.onodera
"""


stats = ['min', 'mean', 'max', 'var']

stats_sum = ['min', 'mean', 'max', 'var', 'sum']

# =============================================================================
# prev
# =============================================================================

prev_num_aggregations = {
    # TODO: optimize stats
    'AMT_ANNUITY': stats,
    'AMT_APPLICATION': stats,
    'AMT_CREDIT': stats,
    'APP_CREDIT_PERC': stats,
    'AMT_DOWN_PAYMENT': stats,
    'AMT_GOODS_PRICE': stats,
    'HOUR_APPR_PROCESS_START': stats,
    'RATE_DOWN_PAYMENT': stats,
    'DAYS_DECISION': stats,
    'CNT_PAYMENT': stats,
    
    'total_debt':                             stats,
    'AMT_CREDIT-d-total_debt':                stats,
    'AMT_GOODS_PRICE-d-total_debt':           stats,
    'AMT_GOODS_PRICE-d-AMT_CREDIT':           stats,
    'AMT_ANNUITY-d-app_AMT_INCOME_TOTAL':     stats,
    'AMT_APPLICATION-d-app_AMT_INCOME_TOTAL': stats,
    'AMT_CREDIT-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL': stats,
    
    'AMT_ANNUITY-d-app_AMT_CREDIT':           stats,
    'AMT_APPLICATION-d-app_AMT_CREDIT':       stats,
    'AMT_CREDIT-d-app_AMT_CREDIT':            stats,
    'AMT_GOODS_PRICE-d-app_AMT_CREDIT':       stats,
    
    'AMT_ANNUITY-d-app_AMT_ANNUITY':          stats,
    'AMT_APPLICATION-d-app_AMT_ANNUITY':      stats,
    'AMT_CREDIT-d-app_AMT_ANNUITY':           stats,
    'AMT_GOODS_PRICE-d-app_AMT_ANNUITY':      stats,
    
    'AMT_ANNUITY-d-app_AMT_GOODS_PRICE':      stats,
    'AMT_APPLICATION-d-app_AMT_GOODS_PRICE':  stats,
    'AMT_CREDIT-d-app_AMT_GOODS_PRICE':       stats,
    'AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE':  stats,
    
    'AMT_ANNUITY-m-app_AMT_INCOME_TOTAL':      stats,
    'AMT_APPLICATION-m-app_AMT_INCOME_TOTAL':      stats,
    'AMT_CREDIT-m-app_AMT_INCOME_TOTAL':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_INCOME_TOTAL':      stats,
    'AMT_ANNUITY-m-app_AMT_CREDIT':      stats,
    'AMT_APPLICATION-m-app_AMT_CREDIT':      stats,
    'AMT_CREDIT-m-app_AMT_CREDIT':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_CREDIT':      stats,
    'AMT_ANNUITY-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_APPLICATION-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_CREDIT-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_ANNUITY-m-app_AMT_ANNUITY':      stats,
    'AMT_APPLICATION-m-app_AMT_ANNUITY':      stats,
    'AMT_CREDIT-m-app_AMT_ANNUITY':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_ANNUITY':      stats,
    'AMT_ANNUITY-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_APPLICATION-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_CREDIT-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_ANNUITY-m-app_AMT_GOODS_PRICE':      stats,
    'AMT_APPLICATION-m-app_AMT_GOODS_PRICE':      stats,
    'AMT_CREDIT-m-app_AMT_GOODS_PRICE':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE':      stats,
    'AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL':      stats,
    'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL':      stats,    
    
    'DAYS_FIRST_DRAWING-m-app_DAYS_BIRTH': stats,
    'DAYS_FIRST_DRAWING-m-app_DAYS_EMPLOYED': stats,
    'DAYS_FIRST_DRAWING-m-app_DAYS_REGISTRATION': stats,
    'DAYS_FIRST_DRAWING-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_FIRST_DRAWING-m-app_DAYS_LAST_PHONE_CHANGE': stats,
    'DAYS_FIRST_DUE-m-app_DAYS_BIRTH': stats,
    'DAYS_FIRST_DUE-m-app_DAYS_EMPLOYED': stats,
    'DAYS_FIRST_DUE-m-app_DAYS_REGISTRATION': stats,
    'DAYS_FIRST_DUE-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_FIRST_DUE-m-app_DAYS_LAST_PHONE_CHANGE': stats,
    'DAYS_LAST_DUE_1ST_VERSION-m-app_DAYS_BIRTH': stats,
    'DAYS_LAST_DUE_1ST_VERSION-m-app_DAYS_EMPLOYED': stats,
    'DAYS_LAST_DUE_1ST_VERSION-m-app_DAYS_REGISTRATION': stats,
    'DAYS_LAST_DUE_1ST_VERSION-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_LAST_DUE_1ST_VERSION-m-app_DAYS_LAST_PHONE_CHANGE': stats,
    'DAYS_LAST_DUE-m-app_DAYS_BIRTH': stats,
    'DAYS_LAST_DUE-m-app_DAYS_EMPLOYED': stats,
    'DAYS_LAST_DUE-m-app_DAYS_REGISTRATION': stats,
    'DAYS_LAST_DUE-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_LAST_DUE-m-app_DAYS_LAST_PHONE_CHANGE': stats,
    'DAYS_TERMINATION-m-app_DAYS_BIRTH': stats,
    'DAYS_TERMINATION-m-app_DAYS_EMPLOYED': stats,
    'DAYS_TERMINATION-m-app_DAYS_REGISTRATION': stats,
    'DAYS_TERMINATION-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_TERMINATION-m-app_DAYS_LAST_PHONE_CHANGE': stats,    
    
    'cnt_paid':   ['min', 'mean', 'max', 'var', 'sum'],
    'cnt_paid_ratio': stats,
    'cnt_unpaid': ['min', 'mean', 'max', 'var', 'sum'],
    'amt_paid':   ['min', 'mean', 'max', 'var', 'sum'],
    'amt_unpaid': ['min', 'mean', 'max', 'var', 'sum'],
    'active':     ['min', 'mean', 'max', 'var', 'sum'],
    'completed':  ['min', 'mean', 'max', 'var', 'sum'],
    
}

# =============================================================================
# POS
# =============================================================================

pos_num_aggregations = {
    # TODO: optimize stats
    'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
    'SK_DPD': ['max', 'mean', 'var'],
    'SK_DPD_DEF': ['max', 'mean', 'var'],
    
    'CNT_INSTALMENT_diff':  stats,
    'CNT_INSTALMENT_ratio': stats,
    
    'SK_DPD_diff':          ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over0':    ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over5':    ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over10':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over15':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over20':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over25':   ['max', 'mean', 'var', 'sum'],
}

# =============================================================================
# ins
# =============================================================================

ins_num_aggregations = {
    # TODO: optimize stats
    'NUM_INSTALMENT_VERSION': ['nunique'],
    
    # app
    'DAYS_ENTRY_PAYMENT-m-app_DAYS_BIRTH': stats,
    'DAYS_ENTRY_PAYMENT-m-app_DAYS_EMPLOYED': stats,
    'DAYS_ENTRY_PAYMENT-m-app_DAYS_REGISTRATION': stats,
    'DAYS_ENTRY_PAYMENT-m-app_DAYS_ID_PUBLISH': stats,
    'DAYS_ENTRY_PAYMENT-m-app_DAYS_LAST_PHONE_CHANGE': stats,
    
    'AMT_PAYMENT-m-app_AMT_INCOME_TOTAL': stats,
    'AMT_PAYMENT-m-app_AMT_CREDIT': stats,
    'AMT_PAYMENT-m-app_AMT_ANNUITY': stats,
    'AMT_PAYMENT-m-app_AMT_GOODS_PRICE': stats,
    
    
    'DPD': ['max', 'mean', 'sum', 'nunique'],
    'DBD': ['max', 'mean', 'sum', 'nunique'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT':    ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    'amt_ratio': stats,
    'amt_delta': stats,
    'days_weighted_delay': ['min', 'max', 'mean', 'sum'],
    'delayed_day_over0': stats_sum,
    'delayed_money_0': stats_sum,
    'delayed_money_ratio_0': stats_sum,
    'not-delayed_day_0': stats_sum,
    'not-delayed_money_0': stats_sum,
    'not-delayed_money_ratio_0': stats_sum,
    'delayed_day_over5': stats_sum,
    'delayed_money_5': stats_sum,
    'delayed_money_ratio_5': stats_sum,
    'not-delayed_day_5': stats_sum,
    'not-delayed_money_5': stats_sum,
    'not-delayed_money_ratio_5': stats_sum,
    'delayed_day_over10': stats_sum,
    'delayed_money_10': stats_sum,
    'delayed_money_ratio_10': stats_sum,
    'not-delayed_day_10': stats_sum,
    'not-delayed_money_10': stats_sum,
    'not-delayed_money_ratio_10': stats_sum,
    'delayed_day_over15': stats_sum,
    'delayed_money_15': stats_sum,
    'delayed_money_ratio_15': stats_sum,
    'not-delayed_day_15': stats_sum,
    'not-delayed_money_15': stats_sum,
    'not-delayed_money_ratio_15': stats_sum,
    'delayed_day_over20': stats_sum,
    'delayed_money_20': stats_sum,
    'delayed_money_ratio_20': stats_sum,
    'not-delayed_day_20': stats_sum,
    'not-delayed_money_20': stats_sum,
    'not-delayed_money_ratio_20': stats_sum,
    'delayed_day_over25': stats_sum,
    'delayed_money_25': stats_sum,
    'delayed_money_ratio_25': stats_sum,
    'not-delayed_day_25': stats_sum,
    'not-delayed_money_25': stats_sum,
    'not-delayed_money_ratio_25': stats_sum,
    'delayed_day_over30': stats_sum,
    'delayed_money_30': stats_sum,
    'delayed_money_ratio_30': stats_sum,
    'not-delayed_day_30': stats_sum,
    'not-delayed_money_30': stats_sum,
    'not-delayed_money_ratio_30': stats_sum,
    'delayed_day_over35': stats_sum,
    'delayed_money_35': stats_sum,
    'delayed_money_ratio_35': stats_sum,
    'not-delayed_day_35': stats_sum,
    'not-delayed_money_35': stats_sum,
    'not-delayed_money_ratio_35': stats_sum,
    'delayed_day_over40': stats_sum,
    'delayed_money_40': stats_sum,
    'delayed_money_ratio_40': stats_sum,
    'not-delayed_day_40': stats_sum,
    'not-delayed_money_40': stats_sum,
    'not-delayed_money_ratio_40': stats_sum,
    'delayed_day_over45':         stats_sum,
    'delayed_money_45':           stats_sum,
    'delayed_money_ratio_45':     stats_sum,
    'not-delayed_day_45':         stats_sum,
    'not-delayed_money_45':       stats_sum,
    'not-delayed_money_ratio_45': stats_sum,
    'NUM_INSTALMENT_ratio':stats,
}


# =============================================================================
# cre
# =============================================================================


cre_num_aggregations = {
#    # TODO: optimize stats
'AMT_BALANCE': stats_sum,
 'AMT_CREDIT_LIMIT_ACTUAL': stats_sum,
 'AMT_DRAWINGS_ATM_CURRENT': stats_sum,
 'AMT_DRAWINGS_CURRENT': stats_sum,
 'AMT_DRAWINGS_OTHER_CURRENT': stats_sum,
 'AMT_DRAWINGS_POS_CURRENT': stats_sum,
 'AMT_INST_MIN_REGULARITY': stats_sum,
 'AMT_PAYMENT_CURRENT': stats_sum,
 'AMT_PAYMENT_TOTAL_CURRENT': stats_sum,
 'AMT_RECEIVABLE_PRINCIPAL': stats_sum,
 'AMT_RECIVABLE': stats_sum,
 'AMT_TOTAL_RECEIVABLE': stats_sum,
 'CNT_DRAWINGS_ATM_CURRENT': stats_sum,
 'CNT_DRAWINGS_CURRENT': stats_sum,
 'CNT_DRAWINGS_OTHER_CURRENT': stats_sum,
 'CNT_DRAWINGS_POS_CURRENT': stats_sum,
 'CNT_INSTALMENT_MATURE_CUM': stats_sum,
 
 'SK_DPD': stats_sum,
 'SK_DPD_DEF': stats_sum,
 
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL': stats_sum,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL': stats_sum,
 'AMT_BALANCE-d-app_AMT_CREDIT': stats_sum,
 'AMT_BALANCE-d-app_AMT_ANNUITY': stats_sum,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE': stats_sum,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT': stats_sum,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL': stats_sum,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL': stats_sum,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT': stats_sum,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY': stats_sum,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE': stats_sum,
 
 'SK_DPD_diff': stats_sum,
 'SK_DPD_diff_over0': stats_sum,
 'SK_DPD_diff_over5': stats_sum,
 'SK_DPD_diff_over10': stats_sum,
 'SK_DPD_diff_over15': stats_sum,
 'SK_DPD_diff_over20': stats_sum,
 'SK_DPD_diff_over25': stats_sum,
}

# =============================================================================
# 
# =============================================================================


bure_num_aggregations = {
        # TODO: optimize stats
        'DAYS_CREDIT': stats_sum,
        'CREDIT_DAY_OVERDUE': stats_sum,
        'DAYS_CREDIT_ENDDATE': stats_sum,
        'DAYS_ENDDATE_FACT': stats_sum,
        'AMT_CREDIT_MAX_OVERDUE': stats_sum,
        'CNT_CREDIT_PROLONG': stats_sum,
        'AMT_CREDIT_SUM': stats_sum,
        'AMT_CREDIT_SUM_DEBT': stats_sum,
        'AMT_CREDIT_SUM_LIMIT': stats_sum,
        'AMT_CREDIT_SUM_OVERDUE': stats_sum,
        'DAYS_CREDIT_UPDATE': stats_sum,
        'AMT_ANNUITY': stats_sum,
        
        # app
        'AMT_CREDIT_SUM-d-app_AMT_INCOME_TOTAL': stats_sum,
        'AMT_CREDIT_SUM-d-app_AMT_CREDIT': stats_sum,
        'AMT_CREDIT_SUM-d-app_AMT_ANNUITY': stats_sum,
        'AMT_CREDIT_SUM-d-app_AMT_GOODS_PRICE': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-app_AMT_INCOME_TOTAL': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-app_AMT_CREDIT': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-app_AMT_ANNUITY': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-app_AMT_GOODS_PRICE': stats_sum,
        'AMT_CREDIT_SUM_LIMIT-d-app_AMT_INCOME_TOTAL': stats_sum,
        'AMT_CREDIT_SUM_LIMIT-d-app_AMT_CREDIT': stats_sum,
        'AMT_CREDIT_SUM_LIMIT-d-app_AMT_ANNUITY': stats_sum,
        'AMT_CREDIT_SUM_LIMIT-d-app_AMT_GOODS_PRICE': stats_sum,
        'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_INCOME_TOTAL': stats_sum,
        'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_CREDIT': stats_sum,
        'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_ANNUITY': stats_sum,
        'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_GOODS_PRICE': stats_sum,
        
        'DAYS_CREDIT-m-app_DAYS_BIRTH': stats_sum,
        'DAYS_CREDIT-m-app_DAYS_EMPLOYED': stats_sum,
        'DAYS_CREDIT-m-app_DAYS_REGISTRATION': stats_sum,
        'DAYS_CREDIT-m-app_DAYS_ID_PUBLISH': stats_sum,
        'DAYS_CREDIT-m-app_DAYS_LAST_PHONE_CHANGE': stats_sum,
        'DAYS_CREDIT_ENDDATE-m-app_DAYS_BIRTH': stats_sum,
        'DAYS_CREDIT_ENDDATE-m-app_DAYS_EMPLOYED': stats_sum,
        'DAYS_CREDIT_ENDDATE-m-app_DAYS_REGISTRATION': stats_sum,
        'DAYS_CREDIT_ENDDATE-m-app_DAYS_ID_PUBLISH': stats_sum,
        'DAYS_CREDIT_ENDDATE-m-app_DAYS_LAST_PHONE_CHANGE': stats_sum,
        'DAYS_ENDDATE_FACT-m-app_DAYS_BIRTH': stats_sum,
        'DAYS_ENDDATE_FACT-m-app_DAYS_EMPLOYED': stats_sum,
        'DAYS_ENDDATE_FACT-m-app_DAYS_REGISTRATION': stats_sum,
        'DAYS_ENDDATE_FACT-m-app_DAYS_ID_PUBLISH': stats_sum,
        'DAYS_ENDDATE_FACT-m-app_DAYS_LAST_PHONE_CHANGE': stats_sum,
        
        
        'DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT': stats_sum,
        'DAYS_ENDDATE_FACT-m-DAYS_CREDIT': stats_sum,
        'DAYS_ENDDATE_FACT-m-DAYS_CREDIT_ENDDATE': stats_sum,
        'DAYS_CREDIT_UPDATE-m-DAYS_CREDIT': stats_sum,
        'DAYS_CREDIT_UPDATE-m-DAYS_CREDIT_ENDDATE': stats_sum,
        'DAYS_CREDIT_UPDATE-m-DAYS_ENDDATE_FACT': stats_sum,
        'AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM': stats_sum,
        'AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT': stats_sum,
        'AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT': stats_sum,
        'AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT': stats_sum,
        'AMT_CREDIT_SUM-d-debt-p-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT': stats_sum,
}




















