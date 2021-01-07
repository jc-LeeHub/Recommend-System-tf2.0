'''
# Time   : 2020/12/17 21:33
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Attention
import tensorflow.keras.backend as K

class Embed_layer(Layer):
    def __init__(self, k, sparse_feature_columns):
        super(Embed_layer, self).__init__()
        self.emb_layers = [Embedding(feat['feat_onehot_dim'], k) for feat in sparse_feature_columns]

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))

        emb = tf.transpose(
                    tf.convert_to_tensor([layer(inputs[:, i])
                                for i, layer in enumerate(self.emb_layers)]),
                    [1, 0, 2])
        emb = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2]))
        return emb

class Res_layer(Layer):
    def __init__(self, hidden_units):
        super(Res_layer, self).__init__()
        self.dense_layer = [Dense(i, activation='relu') for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = Dense(input_shape[-1], activation=None)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))

        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)

        output = inputs + x
        return tf.nn.relu(output)

