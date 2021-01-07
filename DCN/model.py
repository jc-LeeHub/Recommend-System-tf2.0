'''
# Time   : 2020/12/2 11:15
# Author : junchaoli
# File   : model.py
'''

from layer import Dense_layer, Cross_layer

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras import Model

class DCN(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num, reg_w=1e-4, reg_b=1e-4):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dense_layer = Dense_layer(hidden_units, output_dim, activation)
        self.cross_layer = Cross_layer(layer_num, reg_w=reg_w, reg_b=reg_b)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=1)

        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.dense_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
        return output





