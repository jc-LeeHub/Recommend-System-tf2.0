'''
# Time   : 2020/12/11 10:04
# Author : junchaoli
# File   : utils.py
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_criteo_dataset(file_path, test_size=0.3):
    data = pd.read_csv(file_path)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    #缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    #归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    #Onehot编码
    data = pd.get_dummies(data)

    #数据集划分
    X = data.drop(['label'], axis=1).values
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)