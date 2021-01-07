'''
# Time   : 2021/1/4 11:51
# Author : junchaoli
# File   : layer.py
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout

class Dense_layer(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.0):
        super(Dense_layer, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
            x = self.dropout(x)
        return x

class DotProductAttention(Layer):
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = K.softmax(score)                    # SoftMax
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs

class MultiHeadAttention(Layer):
    def __init__(self, n_heads=4, head_dim=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_values')

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads*self._head_dim != queries.shape[-1]:
            raise ValueError("n_head * head_dim not equal embedding dim {}".format(queries.shape[-1]))

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)           # [None, n, k]
        values_linear = K.dot(values, self._weights_values)     # [None, n, k]

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0) # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)       # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)   # [None*n_head, n, k/n_head]

        att_out = self._att_layer([queries_multi_heads, keys_multi_heads, values_multi_heads])   # [None*n_head, n, k/n_head]
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)    # [None, n, k]
        return outputs

