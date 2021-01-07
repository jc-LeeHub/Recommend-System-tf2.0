'''
# Time   : 2020/12/3 21:03
# Author : junchaoli
# File   : model.py
'''

from layer import Dense_layer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, BatchNormalization

class NFM(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation='relu', dropout=0.):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layers = Dense_layer(hidden_units, output_dim, activation, dropout)
        self.emb_layers = {'emb_'+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                           for i, feat in enumerate(self.sparse_feature_columns)}
        self.bn_layer = BatchNormalization()
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = [self.emb_layers['emb_'+str(i)](sparse_inputs[:, i])
               for i in range(sparse_inputs.shape[1])]  # list
        emb = tf.convert_to_tensor(emb)          # (26, None, embed_dim)
        emb = tf.transpose(emb, [1, 0, 2])       # (None, 26, embed_dim)

        # Bi-Interaction Layer
        emb = 0.5 * (tf.pow(tf.reduce_sum(emb, axis=1), 2) -
                       tf.reduce_sum(tf.pow(emb, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = tf.concat([dense_inputs, emb], axis=-1)
        x = self.bn_layer(x)
        x = self.dnn_layers(x)

        outputs = self.output_layer(x)
        return tf.nn.sigmoid(outputs)