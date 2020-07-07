# coding: utf-8
import os, copy, functools
import numpy  as np
import pandas as pd
from datetime import datetime
from IPython.display import display

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def one_hot_encoding(df_train, df_test=None, object_cols=None, isdrop=True):
    """
      This function can do one hot encoding of categorical features.
    """
    
    def get_object_cols(df):
        return list(df.select_dtypes(include="object").columns)

    df_train["train_test"] = "train"
    if df_test is not None:
        df_test["train_test"] = "test"
        df_concat = pd.concat([df_train, df_test]).reset_index(drop=True)
        if object_cols is None: object_cols = list(set(get_object_cols(df_train) + get_object_cols(df_test)))
    else:
        df_concat = df_train.copy().reset_index(drop=True)
        if object_cols is None: object_cols = list(set(get_object_cols(df_train)))

    df_ohe     = pd.get_dummies(df_concat[object_cols], drop_first=True)
    if isdrop:
        df_ohe = pd.merge(df_concat.drop(object_cols, axis=1), df_ohe, left_index=True, right_index=True)
    else:
        df_ohe = pd.merge(df_concat, df_ohe, left_index=True, right_index=True)
        
    if df_test is not None:
        df_ohe_train = df_ohe.query("train_test_train==1").drop("train_test_train", axis=1)
        df_ohe_test  = df_ohe.query("train_test_train==0").drop("train_test_train", axis=1).reset_index(drop=True)
        return df_ohe_train, df_ohe_test
    else:
        return df_ohe

def standardization(df_train, df_test=None, numeric_cols=None):
    """
      This function can do standardization of numerical features.
    """
    
    def get_numeric_cols(df):
        return list(df.select_dtypes(include=["int","float"]).columns)

    if numeric_cols is None:
        if df_test is not None:
            numeric_cols = list(set(get_numeric_cols(df_train) + get_numeric_cols(df_test)))
        else:
            numeric_cols = list(set(get_numeric_cols(df_train)))
    
    mean   = df_train[numeric_cols].mean()
    std    = df_train[numeric_cols].std()
    df_train_std    = df_train.copy()
    df_train_std[numeric_cols]    = df_train_std[numeric_cols].apply(lambda x: (x - mean[x.name]) / std[x.name])
    if df_test is not None:
        df_test_std = df_test.copy()
        df_test_std[numeric_cols] = df_test_std[numeric_cols].apply( lambda x: (x - mean[x.name]) / std[x.name])
        return df_train_std, df_test_std
    else:
        return df_train_std

def create_nan_feature(df, add_row_nan=True, add_is_nan=True):
    """
      This function can add features about Nan.
    """
    
    df_added_nan = df.copy()
    print("The shape before adding features of Nan:", df_added_nan.shape)
    
    if add_row_nan:
        df_added_nan['number_of_nan'] = df_added_nan.isna().sum(axis=1).astype(np.int8)
        
    if add_is_nan:
        for col in df_added_nan.columns:
            if df_added_nan[col].isna().any():
                df_added_nan[col + "_nan"] = np.where(df_added_nan[col].isna(), 1, 0)
            
    print("The shape after  adding features of Nan:", df_added_nan.shape)
    return df_added_nan

def showStats(df):
    """
      This function can show the statistics of features.
      
      Explaination of the result dataframe columns.
        Feature name                      : カラム名
        Unique values                     : カラムごとのユニーク数
        Most frequent item                : 最も出現頻度の高い値
        Freuquence of most frequent item  : 最も出現頻度の高い値の出現回数
        Missing values(%)                 : 欠損損値の割合
        Values in the biggest category(%) : 最も多いカテゴリの割合
        Type                              : 型
    """
    
    stats = []
    for col in df.columns:
        stats.append((col,
                      df[col].nunique(),
                      df[col].value_counts().index[0],
                      df[col].value_counts().values[0],
                      df[col].isnull().sum() * 100 / df.shape[0],
                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                      df[col].dtype))
    df_stats = pd.DataFrame(stats, columns=['Feature name', 'Unique values', 'Most frequent item', 'Freuquence of most frequent item',
                                            'Missing values(%)', 'Values in the biggest category(%)', 'Type'])
    display(df_stats)

def showList(show_list, show_num=50, col=True):
    """
      This function can show reshaped List.
    """
    
    reshaped_list = []
    if show_num < len(show_list):
        for i in range(0, len(show_list)+show_num, show_num):
            if len(show_list) < i:
                break
            l = sorted(show_list)[i:i+show_num]
            if len(l)==show_num:
                reshaped_list.append(l)
            else:
                reshaped_list.append(l + [None]*(show_num-len(l)))
    else:
        reshaped_list = [sorted(show_list)]
        
    df_show_col = pd.DataFrame(reshaped_list)
    if 0 < df_show_col.shape[1]:
        display(df_show_col) if col else display(df_show_col.T)
    else:
        print("No features")

def featureExtractionCorr(df, thrs=[0.99]):
    """
      This function can do feature extraction by correration.
    """
    
    dict_features = {}
    for thr in thrs:
        dict_features[str(thr)] = []
        
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            for thr in thrs:
                if thr < abs(corr_matrix.iloc[i, j]):
                    dict_features[str(thr)].append(corr_matrix.columns[i])
                    
    for key, item in dict_features.items():
        dict_features[key] = sorted(set(dict_features[key]))
        print(key, "The number of features is %s" % len(dict_features[key]))
        
    del corr_matrix
    return dict_features

def reduceMemUsage(df, verbose=False, y=[]):
    """
      This function can reduce memory usage of DataFrame.
    """

    numerics  = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if (col in y) or (col_type not in numerics):
            continue
            
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if   c_min > np.iinfo(np.int8).min  and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
        else:
            if   c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
                
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

