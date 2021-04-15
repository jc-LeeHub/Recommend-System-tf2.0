'''
# Time   : 2020/10/22 15:26
# Author : junchaoli
# File   : train_lstm.py
'''

from model import WideDeep
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file_path = 'E:\\PycharmProjects\\推荐算法\\data\\train.txt'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.2)

    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    model = WideDeep(feature_columns, hidden_units, output_dim, activation)
    optimizer = optimizers.SGD(0.01)

    # train_dataset = tf.data.Dataset.from_tensor_slices(((X_train[:, :13], X_train[:, 13:]), y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(train_dataset, epochs=1000)
    # logloss, auc = model.evaluate((X_test[:, :13], X_test[:, 13:]), y_test)
    # print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))
    # model.summary()
    
    # tensorboard可视化(不需要可以注释掉)
    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        # 可视化
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("ACC: ", accuracy_score(y_test, pre))
