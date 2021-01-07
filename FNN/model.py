'''
# Time   : 2020/12/11 16:11
# Author : junchaoli
# File   : model.py
'''

from layer import FM_layer, DNN_layer
from tensorflow.keras.models import Model

class FM(Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.fm = FM_layer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output

class DNN(Model):
    def __init__(self, hidden_units, output_dim, activation='relu'):
        super().__init__()
        self.dnn = DNN_layer(hidden_units, output_dim, activation)

    def call(self, inputs, training=None, mask=None):
        output = self.dnn(inputs)
        return output