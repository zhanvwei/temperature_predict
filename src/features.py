# -*- encoding: utf-8 -*-
'''
@File    :   features.py
@Time    :   2020/07/27 22:28:09
@Author  :   zhanxv 
@Desc    :   None
'''

# here put the import lib
# 0. 导入相关包
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import featuretools as ft
from sklearn.preprocessing import KBinsDiscretizer

# 1. 读取数据


def read_data(train_path, test_path):
    """
    :type train_path: str
    :type test_path: str
    :rtype : DataFrame
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df.columns = ['tim','year','month','day','hour','min','sec','outdoorTemp','outdoorHum',\
                        'outdoorAtmo','indoorHum','indoorAtmo', 'indoorTemp']
    test_df.columns = ['tim','year','month','day','hour','min','sec','outdoorTemp','outdoorHum',\
                        'outdoorAtmo','indoorHum', 'indoorAtmo']
    return train_df, test_df


# 2. 特征工程

## 2.1 滑窗t时刻近n分钟的温度/湿度/压强


def get_sliding_window_features(df, features, delta):
    """
    :type df: DataFrame
    :type features: list[int]
    :type delta: int
    """
    opts = ['max', 'min', 'median', 'mean', 'std']
    cols = [
        "recent_{}_minute_{}_{}".format(delta, i, j) for i in features
        for j in opts
    ]
    result = pd.DataFrame(columns=['time'] + cols)
    observe = df['time'].values
    for i, obs in enumerate(observe):
        item = __get_sliding_window_features(df, features, obs, delta)
        result.loc[i] = item
    return result


def __get_sliding_window_features(df, features, observe, delta):
    """
    :type df: DataFrame
    :type features: list[str]
    :type observe: timestamp
    :type delta: int
    :rtype : DataFrame
    """
    start = observe - delta * 60
    copy_df = get_sliding_window(df, start, observe)
    item = [observe]
    for f in features:
        item += [np.max(copy_df[f]), np.min(copy_df[f]), np.median(copy_df[f]),\
                 np.mean(copy_df[f]), np.std(copy_df[f])]
    return item


def get_sliding_window(df, start, end, tm='time'):
    """
    :type df: DataFrame
    :type start: timestamp
    :type end: timestamp
    :rtype: DataFrame
    """
    cond = (df[tm] > start) & (df[tm] <= end)
    copy_df = copy.deepcopy(df.loc[cond])
    return copy_df


## 2.2 交叉特征
def get_cross_features(df, features, key='tim'):
    """
    :type df: DataFrame
    :type features: list[str]]
    :rtype: DataFrame
    """
    use_df = copy.deepcopy(df.loc[:, [key] + features])
    es = ft.EntitySet(id='temperature_predict')
    es = es.entity_from_dataframe(entity_id='temp',
                                  dataframe=use_df,
                                  index=key)
    trans_primitives = [
        'add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric'
    ]

    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_entity='temp',
        max_depth=1,  # max_depth=1，只在原特征上进行运算产生新特征
        verbose=1,
        trans_primitives=trans_primitives)
    features_df = pd.DataFrame(feature_matrix).reset_index()
    features_df.drop(columns=features, inplace=True)
    return features_df


### 2.3 分箱特征
def get_bins_features(df, bins, features, key='tim'):
    """
    :type df: DataFrame
    :type bins: int
    """
    use_df = copy.deepcopy(df.loc[:, ['key'] + feautures])
    for f in tqdm(features):
        for b in bins:
             col = f + '_{}_bin'.format(b)
             use_df[col] = pd.cut(use_df[f], bin,
                             duplicates = 'drop').apply(lambda x: x.left).astype(int)
    use_df.drop(columns = features, inplace = True)
    return use_df


### 2.4 聚合特征

def get_group_features(df, features, groups = ['month','day','hour'], key = 'tim'):
    """
    根据keys进行聚合
    :type df: [type]
    :type keys: list, optional
    """
    use_df = copy.deepcopy(df.loc[:, keys + feautures])
    for f in tqdm(features):
         use_df['group_{}_median'.format(f)] = use_df.groupby(groups )[f].transform('median')
         use_df['group_{}_mean'.format(f)] = use_df.groupby(groups )[f].transform('mean')
         use_df['group_{}_max'.format(f)] = use_df.groupby(groups )[f].transform('max')
         use_df['group_{}_min'.format(f)] = use_df.groupby(groups )[f].transform('min')
         use_df['group_{}_std'.format(f)] = use_df.groupby(groups )[f].transform('std')
    use_df.drop(columns = features, inplace = True)
    return use_df










if __name__ == "__main__":
    PATH = "./temperature_predict/input/"
    TRAIN_PATH = PATH + 'train/train.csv'
    TEST_PATH = PATH + 'test/test.csv'

    train, test = read_data(TRAIN_PATH, TEST_PATH)

    # result = get_sliding_window_features(train, ['outdoorTemp','outdoorHum'], delta= 15)
    # print(result.head())
    # cross_feature_df = get_cross_features(train, ['outdoorTemp','outdoorHum'])
    # print(cross_feature_df.head())
    # result.to_csv("./temperature_predict/output/sliding_windows_features.csv", index= False)
    bin_features = get_bins_features(train, 5, ['outdoorTemp', 'outdoorHum'])
    print(bin_features.head)