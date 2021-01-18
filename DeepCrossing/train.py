'''
# Time   : 2020/12/17 22:00
# Author : junchaoli
# File   : train.py
'''

from utils import create_criteo_dataset
from model import DeepCrossing

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file_path = 'E:\\PycharmProjects\\推荐算法\\data\\train.txt'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.2)

    k = 32
    hidden_units = [256, 256]
    res_layer_num = 4

    model = DeepCrossing(feature_columns, k,  hidden_units, res_layer_num)
    optimizer = optimizers.SGD(0.01)

    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(train_dataset, epochs=100)
    # logloss, auc = model.evaluate(X_test, y_test)
    # print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))

    summary = tf.summary.create_file_writer("E:\\PycharmProjects\\tensorboard")
    for i in range(100):
        with tf.GradientTape() as tape:
            pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))
            print(loss.numpy())
        with summary.as_default():
            tf.summary.scalar('loss', loss, i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
