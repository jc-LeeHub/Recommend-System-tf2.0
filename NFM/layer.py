'''
# Time   : 2020/12/3 21:03
# Author : junchaoli
# File   : layer.py
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation='relu', dropout=0.):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.drop_layer = Dropout(rate=dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.drop_layer(x)
        return output
