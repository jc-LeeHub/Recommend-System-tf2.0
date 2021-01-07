'''
# Time   : 2020/12/1 21:41
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.regularizers import l2

class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_num, self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # one-hot encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)
        for i in range(sparse_inputs.shape[1]):
            x = tf.concat(
                [x, tf.one_hot(tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                                   depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)

        linear_part = self.w0 + tf.matmul(x, self.w)
        inter_part = 0
        # 每维特征先跟自己的 [field_num, k] 相乘得到Vij*X
        field_f = tf.tensordot(x, self.v, axes=1)  # [None, 2291] x [2291, 39, 8] = [None, 39, 8]
        # 域之间两两相乘，
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
                    axis=1, keepdims=True
                )

        return linear_part + inter_part

