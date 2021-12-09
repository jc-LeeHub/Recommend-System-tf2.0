'''
# Time   : 2021/12/9 14:47
# Authur : junchaoli
# File   : model.py
'''

from layer import mmoe_layer, tower_layer

import tensorflow as tf
from tensorflow.keras import Model

class MMOE(Model):
    def __init__(self, mmoe_hidden_units, num_experts, num_tasks,
                 tower_hidden_units, output_dim, activation='relu',
                 use_expert_bias=True, use_gate_bias=True,
                 **kwargs):

        super(MMOE, self).__init__()
        self.mmoe_layer = mmoe_layer(mmoe_hidden_units,
                                     num_experts,
                                     num_tasks,
                                     use_expert_bias,
                                     use_gate_bias)

        # 每个任务对应一个tower_layer
        self.tower_layer = [
            tower_layer(tower_hidden_units, output_dim, activation)
            for _ in range(num_tasks)
        ]

    def call(self, inputs):
        mmoe_outputs = self.mmoe_layer(inputs)  # list: num_tasks x [None, hidden_units]

        outputs = []
        for i, layer in enumerate(self.tower_layer):
            out = layer(mmoe_outputs[i])
            outputs.append(out)

        return outputs  # list: num_tasks x [None, output_dim]
