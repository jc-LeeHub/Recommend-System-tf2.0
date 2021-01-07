'''
# Time   : 2020/12/29 11:23
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization

class Attention(Layer):
    def __init__(self, hidden_units, activation='prelu'):
        super(Attention, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        # query: [None, k]
        # key:   [None, n, k]
        # value: [None, n, k]
        # mask:  [None, n]
        query, key, value, mask = inputs

        query = tf.expand_dims(query, axis=1)        # [None, 1, k]
        query = tf.tile(query, [1, key.shape[1], 1]) # [None, n, k]

        emb = tf.concat([query, key, query-k, query*k], axis=-1) # [None, n, 4*k]

        for layer in self.dense_layer:
            emb = layer(emb)
        score = self.out_layer(emb)         # [None, n, 1]
        score = tf.squeeze(score, axis=-1) # [None, n]

        padding = tf.ones_like(score) * (-2**32 + 1)           # [None, n]
        score = tf.where(tf.equal(mask, 0), padding, score)    # [None, n]

        score = tf.nn.softmax(score)
        output = tf.matmul(tf.expand_dims(score, axis=1), value) # [None, 1, k]
        output = tf.squeeze(output, axis=1) # [None, k]
        return output

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn_layer = BatchNormalization()
        self.alpha = self.add_weight(name='alpha', shape=(1,), trainable=True)

    def call(self, inputs, **kwargs):
        x = self.bn_layer(inputs)
        x = tf.nn.sigmoid(x)
        output = x * inputs + (1-x) * self.alpha * inputs
        return output


