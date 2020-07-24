# -*- encoding: utf-8 -*-
'''
@File    :   baseline.py
@Time    :   2020/07/23 14:08:03
@Author  :   zhanxv 
@Desc    :   None
'''

# here put the import lib


### 0. 导入相关包
import pandas as pd
import numpy as np



### 1. 读取数据
train_df =  pd.read_csv("./temperature_predict/input/train/train.csv")
test_df = pd.read_csv("./temperature_predict/input/test/test.csv")


print(train_df.head())
print(test_df.head())
