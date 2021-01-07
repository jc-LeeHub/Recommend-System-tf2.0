'''
# Time   : 2020/12/11 15:54
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

class FM_layer(Layer):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w1 = self.add_weight(name='w1', shape=(input_shape[-1], 1),
                                  initializer=tf.random_normal_initializer(),
                                  trainable=True,
                                  regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                  initializer=tf.random_normal_initializer(),
                                  trainable=True,
                                  regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs, self.w1) + self.w0

        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = tf.reduce_sum(inter_part1-inter_part2, axis=-1, keepdims=True) / 2

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)

class DNN_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu', dropout=0.2):
        super().__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.dropout_layer = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout_layer(x)
        output = self.output_layer(x)
        return tf.nn.sigmoid(output)

