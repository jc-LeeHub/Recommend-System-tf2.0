'''
# Time   : 2021/12/9 17:14
# Authur : junchaoli
# File   : train.py
'''

from model import MMOE
import tensorflow as tf


if __name__ == '__main__':
    mmoe_hidden_units = 10
    num_experts = 3
    num_tasks = 3
    tower_hidden_units = [20, 10]
    tower_output_dim = 1

    model = MMOE(mmoe_hidden_units,
                 num_experts,
                 num_tasks,
                 tower_hidden_units,
                 tower_output_dim)

    # 模拟输入input，shape[2, 4]，两个四维的样本
    input = tf.constant(
        [[1., 1., 1., 3.],
         [2., 2., 1., 4.]],
        dtype=tf.float32
    )

    output = model(input)
    print(output)  # list: num_tasks x [2, output_dim]
