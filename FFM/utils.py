'''
# Time   : 2020/12/1 20:53
# Author : junchaoli
# File   : __init__.py
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

    #缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    #归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    #LabelEncoding编码
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col]).astype(int)

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, data[feat].nunique(), embed_dim) for feat in sparse_features]]

    X = data.drop(['label'], axis=1).values
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)