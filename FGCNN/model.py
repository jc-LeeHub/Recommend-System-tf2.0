'''
# Time   : 2021/1/7 14:20
# Author : junchaoli
# File   : model.py
'''

from layer import DNN, FGCNN_layer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding

class FGCNN(Model):
    def __init__(self, feature_columns, hidden_units, out_dim=1, activation='relu', dropout=0.0,
                 filters=[14, 16], kernel_width=[7, 7], dnn_maps=[3, 3], pooling_width=[2, 2]):
        super(FGCNN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i,feat in enumerate(self.sparse_feature_columns)]
        self.dnn_layer = DNN(hidden_units, out_dim=out_dim, activation=activation, dropout=dropout)
        self.fgcnn_layer = FGCNN_layer(filters=filters, kernel_width=kernel_width, dnn_maps=dnn_maps, pooling_width=pooling_width)

    def call(self, inputs, training=None, mask=None):
        # dense_inputs:  [None, 13]
        # sparse_inputs: [None, 26]
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_embed = [layer(sparse_inputs[:, i]) for i, layer in enumerate(self.emb_layers)] # 26 * [None, k]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])             # [None, 26, k]

        fgcnn_out = self.fgcnn_layer(sparse_embed)          # [None, new_n, k]
        sparse_embed = tf.concat([sparse_embed, fgcnn_out], axis=1)
        sparse_embed = tf.reshape(sparse_embed, shape=[-1, sparse_embed.shape[1]*sparse_embed.shape[2]])

        input = tf.concat([dense_inputs, sparse_embed], axis=-1)
        output = self.dnn_layer(input)
        return output
