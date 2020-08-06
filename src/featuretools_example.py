# -*- encoding: utf-8 -*-
'''
@File    :   featuretools_example.py
@Time    :   2020/07/30 23:39:04
@Author  :   zhanxv 
@Desc    :   None
'''

# here put the import lib

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import featuretools as ft



if __name__ == "__main__":
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    iris_feature_names = dataset.feature_names
    df = pd.DataFrame(X, columns=iris_feature_names)
    es = ft.EntitySet(id='single_dataframe')  # 用id标识实体集
    # 增加一个数据框，命名为iris
    es.entity_from_dataframe(entity_id='iris',
                             dataframe=df,
                             index='index',
                             make_index=True)
    trans_primitives=['add_numeric', 'subtract_numeric', ,'multiply_numeric', 'divide_numeric']  # 2列相加减乘除来生成新特征
    feature_matrix, feature_names = ft.dfs(entityset=es,
                                            target_entity='iris',
                                            max_depth=1,    # max_depth=1，只在原特征上进行运算产生新特征
                                            verbose=1,
                                            trans_primitives=trans_primitives
                                            )
    ft.list_primitives()  # 查看可使用的特征集元
    # features_df = pd.DataFrame(feature_matrix, columns= feature_names)
    # print(features_df.head())
    print(feature_matrix)
 
