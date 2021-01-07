'''
# Time   : 2021/1/4 12:11
# Author : junchaoli
# File   : model.py
'''

from layer import Dense_layer, DotProductAttention, MultiHeadAttention

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding

class AutoInt(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                    dnn_dropout=0.0, n_heads=4, head_dim=64, att_dropout=0.1):
        super(AutoInt, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dense_emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                      for feat in self.dense_feature_columns]
        self.sparse_emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                      for feat in self.sparse_feature_columns]
        self.dense_layer = Dense_layer(hidden_units, activation, dnn_dropout)
        self.multi_head_att = MultiHeadAttention(n_heads, head_dim, att_dropout)
        self.out_layer = Dense(1, activation=None)
        k = self.dense_feature_columns[0]['embed_dim']
        self.W_res = self.add_weight(name='W_res', shape=(k, k),
                                     trainable=True,
                                     initializer=tf.initializers.glorot_normal(),
                                     regularizer=tf.keras.regularizers.l1_l2(1e-5))

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # 值为1.0会使embedding报错
        dense_inputs = tf.where(tf.equal(dense_inputs, 1), 0.9999999, dense_inputs)
        dense_emb = [layer(dense_inputs[:, i]) for i, layer in enumerate(self.dense_emb_layers)]     # [13, None, k]
        sparse_emb = [layer(sparse_inputs[:, i]) for i, layer in enumerate(self.sparse_emb_layers)]  # [26, None, k]
        emb = tf.concat([tf.convert_to_tensor(dense_emb), tf.convert_to_tensor(sparse_emb)], axis=0) # [39, None, k]
        emb = tf.transpose(emb, [1, 0, 2])  # [None, 39, k]

        # DNN
        dnn_input = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2])) # [None, 39*k]
        dnn_out = self.dense_layer(dnn_input)  # [None, out_dim]

        # AutoInt
        att_out = self.multi_head_att([emb, emb, emb]) # [None, 39, k]
        att_out_res = tf.matmul(emb, self.W_res)       # [None, 39, k]
        att_out = att_out + att_out_res
        att_out = tf.reshape(att_out, [-1, att_out.shape[1]*att_out.shape[2]]) # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        output = self.out_layer(x)
        return tf.nn.sigmoid(output)


