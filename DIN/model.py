'''
# Time   : 2020/12/29 14:44
# Author : junchaoli
# File   : model.py
'''

from layer import Attention, Dice

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l2

class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units, ffn_hidden_units,
                 att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0.0):
        super(DIN, self).__init__()
        self.maxlen = maxlen
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        self.other_sparse_num = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_num = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        # other sparse embedding
        self.embed_sparse_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns
                                      if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns
                                      if feat['feat'] in behavior_feature_list]

        self.att_layer = Attention_Layer(att_hidden_units, att_activation)
        self.bn_layer = BatchNormalization(trainable=True)
        self.dense_layer = [Dense(unit, activation=PReLU() if ffn_activation=='prelu' else Dice())\
             for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None):
        # dense_inputs:  empty/(None, dense_num)
        # sparse_inputs: empty/(None, other_sparse_num)
        # history_seq:  (None, n, k)
        # candidate_item: (None, k)
        dense_inputs, sparse_inputs, history_seq, candidate_item = inputs

        # dense & sparse inputs embedding
        other_feat = tf.concat([layer(sparse_inputs[:, i]) for i, layer in enumerate(self.embed_sparse_layers)],
                            axis=-1)
        other_feat = tf.concat([other_feat, dense_inputs], axis=-1)

        # history_seq & candidate_item embedding
        seq_embed = tf.concat([layer(history_seq[:, :, i])
                            for i, layer in enumerate(self.embed_seq_layers)],
                            axis=-1)   # (None, n, k)
        item_embed = tf.concat([layer(candidate_item[:, i])
                            for i, layer in enumerate(self.embed_seq_layers)],
                            axis=-1)   # (None, k)

        # one_hot之后第一维是1的token，为填充的0
        mask = tf.cast(tf.not_equal(history_seq[:, :, 0], 0), dtype=tf.float32)   # (None, n)
        att_emb = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, k)

        # 若其他特征不为empty
        if self.dense_len>0 or self.other_sparse_len>0:
            emb = tf.concat([att_emb, item_embed, other_feat], axis=-1)
        else:
            emb = tf.concat([att_emb, item_embed], axis=-1)

        emb = self.bn_layer(emb)
        for layer in self.dense_layer:
            emb = layer(emb)

        emb = self.dropout(emb)
        output = self.out_layer(emb)
        return tf.nn.sigmoid(output) # (None, 1)