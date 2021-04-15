'''
# Time   : 2020/10/22 15:07
# Author : junchaoli
# File   : model.py
'''

from layer import Wide_layer, Deep_layer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding

class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embedding_layer = {'embed_layer'+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i,feat in enumerate(self.sparse_feature_columns)}

        self.wide = Wide_layer()
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        # onehot_inputs：onehot处理的类别特征(wide侧的输入)
        dense_inputs, sparse_inputs, onehot_inputs = inputs[:, :13], inputs[:, 13:39], inputs[:, 39:]

        # wide部分
        wide_input = tf.concat([dense_inputs, onehot_inputs], axis=-1)
        wide_output = self.wide(wide_input)

        # deep部分
        sparse_embed = tf.concat([self.embedding_layer['embed_layer'+str(i)](sparse_inputs[:, i]) \
                        for i in range(sparse_inputs.shape[-1])], axis=-1)
        deep_output = self.deep(sparse_embed)

        output = tf.nn.sigmoid(0.5*(wide_output + deep_output))
        return output
