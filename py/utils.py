
"""
gsutil -m cp ../feature/* gs://homecredit_ko

gsutil -m rsync -d -r ../feature gs://homecredit_ko

./mc ls gcs/homecredit_ko/

../input/POS_CASH_balance.csv.zip: 
    ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 
    'CNT_INSTALMENT_FUTURE', 'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF']
    
../input/application_train.csv.zip: 
    Shape: (307511, 122)
    ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 
    'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
    'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 
    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 
    'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 
    'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 
    'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 
    'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 
    'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
    'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 
    'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 
    'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 
    'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 
    'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 
    'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 
    'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 
    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 
    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 
    'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 
    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 
    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 
    'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 
    'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 
    'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 
    'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 
    'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 
    'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 
    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 
    'AMT_REQ_CREDIT_BUREAU_YEAR']

../input/application_test.csv.zip: 
    Shape: (48744, 121)

../input/bureau.csv.zip: 
    ['SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 
    'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 
    'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 
    'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 
    'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY']


../input/bureau_balance.csv.zip: 
    ['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']

../input/credit_card_balance.csv.zip: 
    ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE', 
    'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 
    'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 
    'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 
    'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 
    'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 
    'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS', 
    'SK_DPD', 'SK_DPD_DEF']

../input/installments_payments.csv.zip: 
    ['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 
    'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']

../input/previous_application.csv.zip: 
    ['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY', 'AMT_APPLICATION', 
    'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'WEEKDAY_APPR_PROCESS_START', 
    'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 
    'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 
    'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE', 
    'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 
    'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA', 
    'NAME_SELLER_INDUSTRY', 'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 
    'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 
    'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL']

../input/sample_submission.csv.zip:
    ['SK_ID_CURR', 'TARGET']


"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from socket import gethostname
HOSTNAME = gethostname()

from tqdm import tqdm
#from itertools import combinations
from sklearn.model_selection import KFold
from time import time, sleep
from datetime import datetime
from multiprocessing import cpu_count, Pool
import gc

# =============================================================================
# global variables
# =============================================================================

COMPETITION_NAME = 'home-credit-default-risk'

SPLIT_SIZE = 20







# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    send_line(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return

def reset_time():
    global st_time
    st_time = time()
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    send_line(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return

def elapsed_minute():
    return (time() - st_time)/60


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def to_feature(df, path):
    
    if df.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { df.columns[df.columns.duplicated()] }')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(f'{path}_{c}.f')
    return

def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    print(f'shape: {df.shape}')
    
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.p')
    return

def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print(f'reading {path}')
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df

#def to_feathers(df, path, split_size=3, inplace=True):
#    """
#    path = '../output/mydf'
#    
#    wirte '../output/mydf/0.f'
#          '../output/mydf/1.f'
#          '../output/mydf/2.f'
#    
#    """
#    if inplace==True:
#        df.reset_index(drop=True, inplace=True)
#    else:
#        df = df.reset_index(drop=True)
#    gc.collect()
#    mkdir_p(path)
#    
#    kf = KFold(n_splits=split_size)
#    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
#        df.iloc[val_index].to_feather(f'{path}/{i:03d}.f')
#    return
#
#def read_feathers(path, col=None):
#    if col is None:
#        df = pd.concat([pd.read_feather(f) for f in tqdm(sorted(glob(path+'/*')))])
#    else:
#        df = pd.concat([pd.read_feather(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
#    return df

def load_train(col=None):
    if col is None:
        return read_pickles('../data/train')
    else:
        return read_pickles('../data/train', col)

def load_test(col=None):
    if col is None:
        return read_pickles('../data/test')
    else:
        return read_pickles('../data/test', col)

def merge(df, col):
    trte = pd.concat([load_train(col=col), #.drop('TARGET', axis=1), 
                      load_test(col=col)])
    df_ = pd.merge(df, trte, on='SK_ID_CURR', how='left')
    return df_

def check_feature():
    
    sw = False
    files = sorted(glob('../feature/train*.f'))
    for f in files:
        path = f.replace('train_', 'test_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    files = sorted(glob('../feature/test*.f'))
    for f in files:
        path = f.replace('test_', 'train_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    if sw:
        raise Exception('Miising file :(')
    else:
        print('All files exist :)')

# =============================================================================
# 
# =============================================================================
def get_dummies(df):
    """
    binary would be drop_first
    """
    col = df.select_dtypes('O').columns.tolist()
    nunique = df[col].nunique()
    col_binary = nunique[nunique==2].index.tolist()
    [col.remove(c) for c in col_binary]
    df = pd.get_dummies(df, columns=col)
    df = pd.get_dummies(df, columns=col_binary, drop_first=True)
    df.columns = [c.replace(' ', '-') for c in df.columns]
    return df


def reduce_mem_usage(df):
    col_int8 = []
    col_int16 = []
    col_int32 = []
    col_int64 = []
    col_float16 = []
    col_float32 = []
    col_float64 = []
    col_cat = []
    for c in tqdm(df.columns, mininterval=20):
        col_type = df[c].dtype

        if col_type != object:
            c_min = df[c].min()
            c_max = df[c].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    col_int8.append(c)
                    
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    col_int16.append(c)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    col_int32.append(c)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    col_int64.append(c)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    col_float16.append(c)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    col_float32.append(c)
                else:
                    col_float64.append(c)
        else:
            col_cat.append(c)
    
    if len(col_int8)>0:
        df[col_int8] = df[col_int8].astype(np.int8)
    if len(col_int16)>0:
        df[col_int16] = df[col_int16].astype(np.int16)
    if len(col_int32)>0:
        df[col_int32] = df[col_int32].astype(np.int32)
    if len(col_int64)>0:
        df[col_int64] = df[col_int64].astype(np.int64)
    if len(col_float16)>0:
        df[col_float16] = df[col_float16].astype(np.float16)
    if len(col_float32)>0:
        df[col_float32] = df[col_float32].astype(np.float32)
    if len(col_float64)>0:
        df[col_float64] = df[col_float64].astype(np.float64)
    if len(col_cat)>0:
        df[col_cat] = df[col_cat].astype('category')


def to_pkl_gzip(df, path):
    df.to_pickle(path)
    os.system('gzip ' + path)
    os.system('rm ' + path)
    return

def check_var(df, var_limit=0, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            df_ = df
#            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df
        
    var = df_.var()
    col_var0 = var[var<=var_limit].index
    if len(col_var0)>0:
        print(f'remove var<={var_limit}: {col_var0}')
    return col_var0

def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df
    
    corr = df_.corr('pearson').abs() # pearson or spearman
    a, b = np.where(corr>=corr_limit)
    col_corr1 = []
    for a_,b_ in zip(a, b):
        if a_ != b_ and a_ not in col_corr1:
#            print(a_, b_)
            col_corr1.append(b_)
    if len(col_corr1)>0:
        col_corr1 = df.iloc[:,col_corr1].columns
        print(f'remove corr>={corr_limit}: {col_corr1}')
    return col_corr1

def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var==False:
        col_corr1 = check_corr(df, corr_limit=corr_limit, sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return

def __get_use_files__():
    
    return

def get_use_files(prefixes=[], is_train=True):
    
    unused_files  = [f.split('/')[-1] for f in sorted(glob('../feature_unused/*.f'))]
    unused_files += [f.split('/')[-1] for f in sorted(glob('../feature_var0/*.f'))]
    unused_files += [f.split('/')[-1] for f in sorted(glob('../feature_corr1/*.f'))]
    
    if is_train:
        all_files = sorted(glob('../feature/train*.f'))
        unused_files = ['../feature/train_'+f for f in unused_files]
    else:
        all_files = sorted(glob('../feature/test*.f'))
        unused_files = ['../feature/test_'+f for f in unused_files]
    
    if len(prefixes)>0:
        use_files = []
        for prefix in prefixes:
            use_files += glob(f'../feature/*{prefix}*')
        all_files = (set(all_files) & set(use_files)) - set(unused_files)
        
    else:
        for f in unused_files:
            if f in all_files:
                all_files.remove(f)
    
    all_files = sorted(all_files)
    
    print(f'got {len(all_files)}')
    return all_files


# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API'):
    os.system(f'kaggle competitions submit -c {COMPETITION_NAME} -f {file_path} -m "{comment}"')
    sleep(60) # tekito~~~~
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'
#        print(f'{i}: {j}') # TODO: comment out later?
    send_line(message.rstrip())

import requests
def send_line(message):
    
    line_notify_token = '5p5sPTY7PrQaB8Wnwp6aadfiqC8m2zh6Q8llrfNisGT'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)

def stop_instance():
    """
    You need to login first.
    >> gcloud auth login
    """
    send_line('stop instance')
    os.system(f'gcloud compute instances stop {os.uname()[1]} --zone us-east1-b')
    
    
