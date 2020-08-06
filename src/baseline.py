# -*- coding:utf-8 -*-
'''
@File    :   baseline.py
@Time    :   2020/07/23 14:08:03
@Author  :   zhanxv 
@Desc    :   None
'''

# here put the import lib


# 0. 导入相关包
import pandas as pd
import numpy as np
import copy
import tqdm

# 1. 读取数据

def read_data(train_path, test_path):
    """
    :type train_path: str
    :type test_path: str
    :rtype : DataFrame
    """
    train_df =  pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum',\
                        'outdoorAtmo','indoorHum','indoorAtmo', 'temperature']
    test_df.columns = ['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum',\
                        'outdoorAtmo','indoorHum', 'indoorAtmo']
    return train_df, test_df



    
    








# 3. 模型训练
def single_model(clf, train_x, train_y, test_x, clf_name, class_num=1):

    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))

    nums = int(train_x.shape[0] * 0.80)

    if clf_name in ['sgd','ridge']:
        print('MinMaxScaler...')
        for col in features:
            ss = MinMaxScaler()
            ss.fit(np.vstack([train_x[[col]].values, test_x[[col]].values]))
            train_x[col] = ss.transform(train_x[[col]].values).flatten()
            test_x[col] = ss.transform(test_x[[col]].values).flatten()

    trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]
    trn_y = np.log1p(trn_y+0.2)
    val_y = np.log1p(val_y+0.2)

    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)
        data_matrix  = clf.Dataset(train_x, label=train_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'mae',
            'min_child_weight': 5,
            'num_leaves': 2 ** 8,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'learning_rate': 0.001,
            'seed': 2020
        }

        model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,early_stopping_rounds=1000)
        model2 = clf.train(params, data_matrix, model.best_iteration)
        val_pred = model.predict(val_x, num_iteration=model2.best_iteration).reshape(-1,1)
        test_pred = model.predict(test_x, num_iteration=model2.best_iteration).reshape(-1,1)

    if clf_name == "xgb":
        train_matrix = clf.DMatrix(trn_x , label=trn_y, missing=np.nan)
        valid_matrix = clf.DMatrix(val_x , label=val_y, missing=np.nan)
        test_matrix  = clf.DMatrix(test_x, label=val_y, missing=np.nan)
        params = {'booster': 'gbtree',
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 8,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'eta': 0.001,
                  'seed': 2020,
                  'nthread': 36,
                  'silent': True,
                  }

        watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500, early_stopping_rounds=1000)
        val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1,1)
        test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit).reshape(-1,1)

    if clf_name == "cat":
        params = {'learning_rate': 0.001, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=20000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=[], use_best_model=True, verbose=500)

        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "sgd":
        params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
        model = SGDRegressor(**params)
        model.fit(trn_x, trn_y)
        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "ridge":
        params = {
                'alpha': 1.0,
                'random_state': 2020,
            }
        model = Ridge(**params)
        model.fit(trn_x, trn_y)
        val_pred  = model.predict(val_x)
        test_pred = model.predict(test_x)


    val_pred = np.expm1(val_pred) -0.2
    test_pred = np.expm1(test_pred) -0.2
    val_y = np.expm1(val_y) -0.2
    trn_y = np.expm1(trn_y) -0.2
    print("%s_mse_score:" % clf_name, mean_squared_error(val_y, val_pred))
    return val_pred, test_pred


def lgb_model(x_train, y_train, x_valid):
    lgb_train, lgb_test = single_model(lgb, x_train, y_train, x_valid, "lgb", 1)
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test

def sgd_model(x_train, y_train, x_valid):
    sgd_train, sgd_test = single_model(SGDRegressor, x_train, y_train, x_valid, "sgd", 1)
    return sgd_train, sgd_test

def ridge_model(x_train, y_train, x_valid):
    ridge_train, ridge_test = single_model(Ridge, x_train, y_train, x_valid, "ridge", 1)
    return ridge_train, ridge_test


drop_columns=["time","year","sec","temperature"]


train_count = train_df.shape[0]
train_df = data_df[:train_count].copy().reset_index(drop=True)
test_df = data_df[train_count:].copy().reset_index(drop=True)

features = train_df[:1].drop(drop_columns,axis=1).columns
x_train = train_df[features]
x_test = test_df[features]

y_train = (train_df['temperature'].values - train_df['outdoorTemp'].values)/train_df['temperature'].values


lr_train, lr_test = ridge_model(x_train, y_train, x_test)

sgd_train, sgd_test = sgd_model(x_train, y_train, x_test)

lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)

xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)

cat_train, cat_test = cat_model(x_train, y_train, x_test)


train_pred = (lr_train + sgd_train + lgb_train[:,0] + xgb_train[:,0] + cat_train) / 5
test_pred = (lr_test + sgd_test + lgb_test[:,0] + xgb_test[:,0] + cat_test) / 5




## 4. 结果保存
sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - xgb_test[:,0])
sub.to_csv('./submission/sub_xgb.csv', index=False)

sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - cat_test)
sub.to_csv('./submission/sub_cat.csv', index=False)

sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - lgb_test[:,0])
sub.to_csv('./submission/sub_lgb.csv', index=False)

sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - sgd_test)
sub.to_csv('./submission/sub_sgd.csv', index=False)

sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - lr_test)
sub.to_csv('./submission/sub_lr.csv', index=False)

sub["temperature"] = test_df['outdoorTemp'].values/(np.ones_like(test_df['outdoorTemp'].values) - test_pred)
sub.to_csv('./submission/sub_all.csv', index=False)
