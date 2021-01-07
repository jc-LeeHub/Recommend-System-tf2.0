'''
# Time   : 2020/12/9 16:31
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, Dropout

class Interaction_layer(Layer):
    '''
    # input shape:  [None, field, k]
    # output shape: [None, field*(field-1)/2, k]
    '''
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs): # [None, field, k]
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        element_wise_product_list = []
        for i in range(inputs.shape[1]):
            for j in range(i+1, inputs.shape[1]):
                element_wise_product_list.append(tf.multiply(inputs[:, i], inputs[:, j]))  #[t, None, k]
        element_wise_product = tf.transpose(tf.convert_to_tensor(element_wise_product_list), [1, 0, 2]) #[None, t, k]
        return element_wise_product

class Attention_layer(Layer):
    '''
    # input shape:  [None, n, k]
    # output shape: [None, k]
    '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape): # [None, field, k]
        self.attention_w = Dense(input_shape[1], activation='relu')
        self.attention_h = Dense(1, activation=None)

    def call(self, inputs, **kwargs): # [None, field, k]
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        x = self.attention_w(inputs)  # [None, field, field]
        x = self.attention_h(x)       # [None, field, 1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1]) # [None, 1, field]
        output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # (None, k)
        return output

class AFM_layer(Layer):
    def __init__(self, feature_columns, mode):
        super(AFM_layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layer = {"emb_"+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                            for i, feat in enumerate(self.sparse_feature_columns)}
        self.interaction_layer = Interaction_layer()
        if self.mode=='att':
            self.attention_layer = Attention_layer()
        self.output_layer = Dense(1)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        embed = [self.embed_layer['emb_'+str(i)](sparse_inputs[:, i])
               for i in range(sparse_inputs.shape[1])]  # list
        embed = tf.convert_to_tensor(embed)
        embed = tf.transpose(embed, [1, 0, 2])  #[None, 26，k]

        # Pair-wise Interaction
        embed = self.interaction_layer(embed)

        if self.mode == 'avg':
            x = tf.reduce_mean(embed, axis=1)  # (None, k)
        elif self.mode == 'max':
            x = tf.reduce_max(embed, axis=1)  # (None, k)
        else:
            x = self.attention_layer(embed)  # (None, k)

        output = tf.nn.sigmoid(self.output_layer(x))
        return output

