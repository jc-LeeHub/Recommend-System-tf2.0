'''
# Time   : 2020/10/15 10:04
# Author : junchaoli
# File   : utils.py
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def create_criteo_dataset(file_path, embed_dim=8, test_size=0.2):
    data = pd.read_csv(file_path)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    y = data['label']
    X = data.drop(['label'], axis=1)

    #缺失值填充
    X[dense_features] = X[dense_features].fillna(0)
    X[sparse_features] = X[sparse_features].fillna('-1')

    #归一化
    X[dense_features] = MinMaxScaler().fit_transform(X[dense_features])

    #Onehot编码(wide侧输入)
    onehot_data = pd.get_dummies(X)

    #LabelEncoding编码(deep侧输入)
    for col in sparse_features:
        X[col] = LabelEncoder().fit_transform(X[col])

    # 拼接到数据集供wide使用
    X = pd.concat([X, onehot_data], axis=1)

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, X[feat].nunique(), embed_dim) for feat in sparse_features]]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)
