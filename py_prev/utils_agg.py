#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:27:07 2018

@author: kazuki.onodera
"""


stats = ['min', 'mean', 'max', 'var']

stats_sum = stats + ['sum']

# =============================================================================
# POS
# =============================================================================

pos_num_aggregations = {
    # TODO: optimize stats
    'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
    'SK_DPD': ['max', 'mean', 'var'],
    'SK_DPD_DEF': ['max', 'mean', 'var'],
    
    'CNT_INSTALMENT-m-CNT_INSTALMENT_FUTURE':  stats,
    'CNT_INSTALMENT_FUTURE-d-CNT_INSTALMENT': stats,
    
     # diff
     'SK_DPD-m-SK_DPD_DEF':          ['max', 'mean', 'var', 'sum'],
     'CNT_INSTALMENT_FUTURE_diff': stats,
     'CNT_INSTALMENT_FUTURE_pctchange': stats,
     'SK_DPD_diff': stats,
     'SK_DPD_pctchange': stats,
     'SK_DPD_DEF_diff': stats,
     'SK_DPD_DEF_pctchange': stats,
#    'SK_DPD_diff_over0':    ['max', 'mean', 'var', 'sum'],
#    'SK_DPD_diff_over5':    ['max', 'mean', 'var', 'sum'],
#    'SK_DPD_diff_over10':   ['max', 'mean', 'var', 'sum'],
#    'SK_DPD_diff_over15':   ['max', 'mean', 'var', 'sum'],
#    'SK_DPD_diff_over20':   ['max', 'mean', 'var', 'sum'],
#    'SK_DPD_diff_over25':   ['max', 'mean', 'var', 'sum'],
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
    
    'AMT_PAYMENT-d-app_AMT_INCOME_TOTAL': stats,
    'AMT_PAYMENT-d-app_AMT_CREDIT': stats,
    'AMT_PAYMENT-d-app_AMT_ANNUITY': stats,
    'AMT_PAYMENT-d-app_AMT_GOODS_PRICE': stats,
    
    # prev
    'NUM_INSTALMENT_ratio': stats,
    'AMT_PAYMENT-d-AMT_ANNUITY': stats,
    
    'DPD': ['max', 'mean', 'sum', 'nunique'],
    'DBD': ['max', 'mean', 'sum', 'nunique'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT':    ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    'amt_ratio': stats,
    'amt_delta': stats,
    'days_weighted_delay': ['min', 'max', 'mean', 'sum'],
    
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
 
 'SK_DPD-m-SK_DPD_DEF': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over0': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over5': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over10': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over15': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over20': stats_sum,
 'SK_DPD-m-SK_DPD_DEF_over25': stats_sum,
 
 # diff
'AMT_BALANCE_diff': stats,
 'AMT_BALANCE_pctchange': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_diff': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_pctchange': stats,
 'AMT_DRAWINGS_ATM_CURRENT_diff': stats,
 'AMT_DRAWINGS_ATM_CURRENT_pctchange': stats,
 'AMT_DRAWINGS_CURRENT_diff': stats,
 'AMT_DRAWINGS_CURRENT_pctchange': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_diff': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_pctchange': stats,
 'AMT_DRAWINGS_POS_CURRENT_diff': stats,
 'AMT_DRAWINGS_POS_CURRENT_pctchange': stats,
 'AMT_INST_MIN_REGULARITY_diff': stats,
 'AMT_INST_MIN_REGULARITY_pctchange': stats,
 'AMT_PAYMENT_CURRENT_diff': stats,
 'AMT_PAYMENT_CURRENT_pctchange': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_diff': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_pctchange': stats,
 'AMT_RECEIVABLE_PRINCIPAL_diff': stats,
 'AMT_RECEIVABLE_PRINCIPAL_pctchange': stats,
 'AMT_RECIVABLE_diff': stats,
 'AMT_RECIVABLE_pctchange': stats,
 'AMT_TOTAL_RECEIVABLE_diff': stats,
 'AMT_TOTAL_RECEIVABLE_pctchange': stats,
 'CNT_DRAWINGS_ATM_CURRENT_diff': stats,
 'CNT_DRAWINGS_ATM_CURRENT_pctchange': stats,
 'CNT_DRAWINGS_CURRENT_diff': stats,
 'CNT_DRAWINGS_CURRENT_pctchange': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_diff': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_pctchange': stats,
 'CNT_DRAWINGS_POS_CURRENT_diff': stats,
 'CNT_DRAWINGS_POS_CURRENT_pctchange': stats,
 'CNT_INSTALMENT_MATURE_CUM_diff': stats,
 'CNT_INSTALMENT_MATURE_CUM_pctchange': stats,
 'SK_DPD_diff': stats,
 'SK_DPD_pctchange': stats,
 'SK_DPD_DEF_diff': stats,
 'SK_DPD_DEF_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_diff': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_diff': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_diff': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_diff': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_pctchange': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_diff': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_diff': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange': stats, 
 
 # diff diff
'AMT_BALANCE_diff_diff': stats,
 'AMT_BALANCE_diff_pctchange': stats,
 'AMT_BALANCE_pctchange_diff': stats,
 'AMT_BALANCE_pctchange_pctchange': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_diff_diff': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_diff_pctchange': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_pctchange_diff': stats,
 'AMT_CREDIT_LIMIT_ACTUAL_pctchange_pctchange': stats,
 'AMT_DRAWINGS_ATM_CURRENT_diff_diff': stats,
 'AMT_DRAWINGS_ATM_CURRENT_diff_pctchange': stats,
 'AMT_DRAWINGS_ATM_CURRENT_pctchange_diff': stats,
 'AMT_DRAWINGS_ATM_CURRENT_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT_pctchange_pctchange': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_diff_diff': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_diff_pctchange': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_pctchange_diff': stats,
 'AMT_DRAWINGS_OTHER_CURRENT_pctchange_pctchange': stats,
 'AMT_DRAWINGS_POS_CURRENT_diff_diff': stats,
 'AMT_DRAWINGS_POS_CURRENT_diff_pctchange': stats,
 'AMT_DRAWINGS_POS_CURRENT_pctchange_diff': stats,
 'AMT_DRAWINGS_POS_CURRENT_pctchange_pctchange': stats,
 'AMT_INST_MIN_REGULARITY_diff_diff': stats,
 'AMT_INST_MIN_REGULARITY_diff_pctchange': stats,
 'AMT_INST_MIN_REGULARITY_pctchange_diff': stats,
 'AMT_INST_MIN_REGULARITY_pctchange_pctchange': stats,
 'AMT_PAYMENT_CURRENT_diff_diff': stats,
 'AMT_PAYMENT_CURRENT_diff_pctchange': stats,
 'AMT_PAYMENT_CURRENT_pctchange_diff': stats,
 'AMT_PAYMENT_CURRENT_pctchange_pctchange': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_diff_diff': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_diff_pctchange': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_pctchange_diff': stats,
 'AMT_PAYMENT_TOTAL_CURRENT_pctchange_pctchange': stats,
 'AMT_RECEIVABLE_PRINCIPAL_diff_diff': stats,
 'AMT_RECEIVABLE_PRINCIPAL_diff_pctchange': stats,
 'AMT_RECEIVABLE_PRINCIPAL_pctchange_diff': stats,
 'AMT_RECEIVABLE_PRINCIPAL_pctchange_pctchange': stats,
 'AMT_RECIVABLE_diff_diff': stats,
 'AMT_RECIVABLE_diff_pctchange': stats,
 'AMT_RECIVABLE_pctchange_diff': stats,
 'AMT_RECIVABLE_pctchange_pctchange': stats,
 'AMT_TOTAL_RECEIVABLE_diff_diff': stats,
 'AMT_TOTAL_RECEIVABLE_diff_pctchange': stats,
 'AMT_TOTAL_RECEIVABLE_pctchange_diff': stats,
 'AMT_TOTAL_RECEIVABLE_pctchange_pctchange': stats,
 'CNT_DRAWINGS_ATM_CURRENT_diff_diff': stats,
 'CNT_DRAWINGS_ATM_CURRENT_diff_pctchange': stats,
 'CNT_DRAWINGS_ATM_CURRENT_pctchange_diff': stats,
 'CNT_DRAWINGS_ATM_CURRENT_pctchange_pctchange': stats,
 'CNT_DRAWINGS_CURRENT_diff_diff': stats,
 'CNT_DRAWINGS_CURRENT_diff_pctchange': stats,
 'CNT_DRAWINGS_CURRENT_pctchange_diff': stats,
 'CNT_DRAWINGS_CURRENT_pctchange_pctchange': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_diff_diff': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_diff_pctchange': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_pctchange_diff': stats,
 'CNT_DRAWINGS_OTHER_CURRENT_pctchange_pctchange': stats,
 'CNT_DRAWINGS_POS_CURRENT_diff_diff': stats,
 'CNT_DRAWINGS_POS_CURRENT_diff_pctchange': stats,
 'CNT_DRAWINGS_POS_CURRENT_pctchange_diff': stats,
 'CNT_DRAWINGS_POS_CURRENT_pctchange_pctchange': stats,
 'CNT_INSTALMENT_MATURE_CUM_diff_diff': stats,
 'CNT_INSTALMENT_MATURE_CUM_diff_pctchange': stats,
 'CNT_INSTALMENT_MATURE_CUM_pctchange_diff': stats,
 'CNT_INSTALMENT_MATURE_CUM_pctchange_pctchange': stats,
 'SK_DPD_diff_diff': stats,
 'SK_DPD_diff_pctchange': stats,
 'SK_DPD_pctchange_diff': stats,
 'SK_DPD_pctchange_pctchange': stats,
 'SK_DPD_DEF_diff_diff': stats,
 'SK_DPD_DEF_diff_pctchange': stats,
 'SK_DPD_DEF_pctchange_diff': stats,
 'SK_DPD_DEF_pctchange_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_diff_diff': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_diff_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_pctchange_diff': stats,
 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL_pctchange_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_diff_diff': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_diff_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_pctchange_diff': stats,
 'AMT_BALANCE-d-app_AMT_CREDIT_pctchange_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_diff_diff': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_diff_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_pctchange_diff': stats,
 'AMT_BALANCE-d-app_AMT_ANNUITY_pctchange_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_diff_diff': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_diff_pctchange': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_pctchange_diff': stats,
 'AMT_BALANCE-d-app_AMT_GOODS_PRICE_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE_pctchange_pctchange': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_diff_diff': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_diff_pctchange': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange_diff': stats,
 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange_pctchange': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_diff_diff': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_diff_pctchange': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_pctchange_diff': stats,
 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT_pctchange_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_diff_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_diff_pctchange': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange_diff': stats,
 'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL_pctchange_pctchange': stats,
 }



















