'''
# Time   : 2021/1/7 13:07
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten, Conv2D, MaxPool2D
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

class FGCNN_layer(Layer):
    def __init__(self, filters=[14, 16], kernel_width=[7, 7], dnn_maps=[3, 3], pooling_width=[2, 2]):
        super(FGCNN_layer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width
        self.dnn_maps = dnn_maps
        self.pooling_width = pooling_width

    def build(self, input_shape):
        # input_shape: [None, n, k]
        n = input_shape[1]
        k = input_shape[-1]
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []
        for i in range(len(self.filters)):
            self.conv_layers.append(
                Conv2D(filters=self.filters[i],
                       kernel_size=(self.kernel_width[i], 1),
                       strides=(1, 1),
                       padding='same',
                       activation='tanh')
            )
            self.pool_layers.append(
                MaxPool2D(pool_size=(self.pooling_width[i], 1))
            )
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        dnn_output = []
        x = tf.expand_dims(inputs, axis=-1) # [None, n, k, 1]最后一维为通道
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)      # [None, n, k, filters[i]]
            x = self.pool_layers[i](x)      # [None, n/poolwidth[i], k, filters[i]]
            out = self.flatten_layer(x)
            out = Dense(self.dnn_maps[i]*x.shape[1]*x.shape[2], activation='relu')(out)
            out = tf.reshape(out, shape=(-1, out.shape[1]//k, k))
            dnn_output.append(out)
        output = tf.concat(dnn_output, axis=1) # [None, new_N, k]
        return output











