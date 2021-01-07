'''
# Time   : 2020/10/22 11:15
# Author : junchaoli
# File   : model.py
'''
from layer import FM_layer, Dense_layer

import tensorflow as tf
from tensorflow.keras import Model

class DeepFM(Model):
    def __init__(self, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.FM = FM_layer(k, w_reg, v_reg)
        self.Dense = Dense_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        fm_output = self.FM(inputs)
        dense_output = self.Dense(inputs)
        output = tf.nn.sigmoid(0.5*(fm_output + dense_output))
        return output
