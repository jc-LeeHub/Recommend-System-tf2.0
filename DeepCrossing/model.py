'''
# Time   : 2020/12/17 21:51
# Author : junchaoli
# File   : model.py
'''

from layer import Embed_layer, Res_layer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class DeepCrossing(Model):
    def __init__(self, feature_columns, k, hidden_units, res_layer_num):
        super(DeepCrossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = Embed_layer(k, self.sparse_feature_columns)
        self.res_layer = [Res_layer(hidden_units) for _ in range(res_layer_num)]
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        emb = self.embed_layer(sparse_inputs)

        x = tf.concat([dense_inputs, emb], axis=-1)

        for layer in self.res_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


