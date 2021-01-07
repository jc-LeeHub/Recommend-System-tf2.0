'''
# Time   : 2021/1/3 15:10
# Author : junchaoli
# File   : model.py
'''

from layer import Linear, Dense_layer, CIN

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding

class xDeepFM(Model):
    def __init__(self, feature_columns, cin_size, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns]
        self.linear = Linear()
        self.dense_layer = Dense_layer(hidden_units, out_dim, activation, dropout)
        self.cin_layer = CIN(cin_size)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # linear
        linear_out = self.linear(inputs)

        emb = [self.embed_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])] # [n, None, k]
        emb = tf.transpose(tf.convert_to_tensor(emb), [1, 0, 2]) # [None, n, k]

        # CIN
        cin_out = self.cin_layer(emb)

        # dense
        emb = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2]))
        emb = tf.concat([dense_inputs, emb], axis=1)
        dense_out = self.dense_layer(emb)

        output = self.out_layer(linear_out + cin_out + dense_out)
        return tf.nn.sigmoid(output)




