'''
# Time   : 2020/12/2 11:15
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [Dense(x, activation=activation) for x in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output

class Cross_layer(Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super().__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b'+str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # (None, dim, 1)
        xl = x0  # (None, dim, 1)
        for i in range(self.layer_num):
            # 先乘后两项（忽略第一维，(dim, 1)表示一个样本的特征）
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i]) # (None, 1, 1)
            # # 乘x0，再加上b、xl
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl  # (None, dim, 1)

        output = tf.squeeze(xl, axis=2)  # (None, dim)
        return output