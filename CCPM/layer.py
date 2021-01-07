'''
# Time   : 2021/1/7 16:51
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten, Conv2D
import tensorflow.keras.backend as K

class DNN(Layer):
    def __init__(self, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(DNN, self).__init__()
        self.dnn_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.drop_layer = Dropout(dropout)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs)!=2:
            raise ValueError("Input dim is not 2 but {}".format(K.ndim(inputs)))
        x = inputs
        for layer in self.dnn_layer:
            x = layer(x)
            x = self.drop_layer(x)
        output = self.out_layer(x)
        return tf.nn.sigmoid(output)

class KMaxPool(Layer):
    def __init__(self, k):
        super(KMaxPool, self).__init__()
        self.k = k

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k, 1]
        inputs = tf.transpose(inputs, [0, 3, 2, 1])
        k_max = tf.nn.top_k(inputs, k=self.k, sorted=True)[0]
        output = tf.transpose(k_max, [0, 3, 2, 1])
        return output

class CCPM_layer(Layer):
    def __init__(self, filters=[4, 4], kernel_width=[6, 5]):
        super(CCPM_layer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width

    def build(self, input_shape):
        n = input_shape[1]
        l = len(self.filters)
        self.conv_layers = []
        self.kmax_layers = []
        for i in range(1, l+1):
            self.conv_layers.append(
                Conv2D(filters=self.filters[i-1],
                       kernel_size=(self.kernel_width[i-1], 1),
                       strides=(1, 1),
                       padding='same',
                       activation='tanh')
            )
            k = max(1, int((1-pow(i/l, l-i))*n)) if i<l else 3 # 论文中k随层数衰减
            self.kmax_layers.append(KMaxPool(k=k))
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        x = tf.expand_dims(inputs, axis=-1) # [None, n, k, 1]
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)      # [None, n, k, filters]
            x = self.kmax_layers[i](x)      # [None, n_k, k, filters]
        output = self.flatten_layer(x)      # [None, n_k*k*filters]
        return output

