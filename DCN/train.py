'''
# Time   : 2020/12/2 11:16
# Author : junchaoli
# File   : train.py
'''
from model import DCN
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file = 'E:\\PycharmProjects\\推荐算法\\data\\train.txt'
    test_size = 0.4
    hidden_units = [256, 128, 64]

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file,
                                           test_size=test_size)

    model = DCN(feature_columns, hidden_units, 1, activation='relu', layer_num=6)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(train_dataset, epochs=100)
    # logloss, auc = model.evaluate(X_test, y_test)
    # print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))
    # model.summary()

    summary_writer = tf.summary.create_file_writer('E:\\PycharmProjects\\tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("Acc: ", accuracy_score(y_test, pre))

