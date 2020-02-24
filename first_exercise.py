# -*- coding: utf-8 -*-
"""https://www.tensorflow.org/install/docker
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


def main():
    # import datasets
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert the bytes to floating point values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # build a keras.sequential model: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    print('ok')
    return


if __name__ == '__main__':
    main()