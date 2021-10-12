'''
# Time   : 2020/12/15 18:55
# Author : junchaoli
# File   : model.py
'''

from layer import DNN_layer, InnerProductLayer, OuterProductLayer, FGCNN_layer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding

class PNN(Model):
    def __init__(self, feature_columns, mode, hidden_units, output_dim, activation='relu', dropout=0.2, use_fgcnn=False):
        super().__init__()
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layer = DNN_layer(hidden_units, output_dim, activation, dropout)
        self.inner_product_layer = InnerProductLayer()
        self.outer_product_layer = OuterProductLayer()
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.use_fgcnn = use_fgcnn
        if use_fgcnn:
            self.fgcnn_layer = FGCNN_layer()

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # sparse inputs embedding
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                          for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])  #[None, field, k]

        # product之前加入fgcnn层
        if self.use_fgcnn:
            fgcnn_out = self.fgcnn_layer(embed)
            embed = tf.concat([embed, fgcnn_out], axis=1)

        z = embed  #[None, field, k]
        embed = tf.reshape(embed, shape=(-1, embed.shape[1]*embed.shape[2]))  # [None, field*k]
        # inner product
        if self.mode=='inner':
            inner_product = self.inner_product_layer(z)   # [None, field*(field-1)/2]
            inputs = tf.concat([embed, inner_product], axis=1)
        # outer product
        elif self.mode=='outer':
            outer_product = self.outer_product_layer(z)   # [None, field*(field-1)/2]
            inputs = tf.concat([embed, outer_product], axis=1)
        # inner and outer product
        elif self.mode=='both':
            inner_product = self.inner_product_layer(z)   # [None, field*(field-1)/2]
            outer_product = self.outer_product_layer(z)   # [None, field*(field-1)/2]
            inputs = tf.concat([embed, inner_product, outer_product], axis=1)
        # Wrong Input
        else:
            raise ValueError("Please choice mode's value in 'inner' 'outer' 'both'.")

        output = self.dnn_layer(inputs)
        return output




