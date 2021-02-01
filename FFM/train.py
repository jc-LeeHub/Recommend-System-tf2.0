'''
# Time   : 2020/12/1 21:41
# Author : junchaoli
# File   : train.py
'''
from model import FFM
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file = 'train.txt'
    test_size = 0.2
    k = 8

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file,
                                           test_size=test_size)

    model = FFM(feature_columns, k=k)
    optimizer = optimizers.SGD(0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
