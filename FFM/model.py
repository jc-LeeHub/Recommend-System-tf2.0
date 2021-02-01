'''
# Time   : 2020/12/1 21:41
# Author : junchaoli
# File   : model.py
'''

from layer import FFM_Layer

import tensorflow as tf
from tensorflow.keras import Model

class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output
