# -*- coding: utf-8 -*-
"""https://www.tensorflow.org/install/docker
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # tell tf that
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       # there's no CUDA here
import datetime

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

    # for each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
    predictions = model(x_train[:1]).numpy()

    # probabilities
    probabilities = tf.nn.softmax(predictions).numpy()

    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_loss = loss_fn(y_train[:1], predictions).numpy()

    # compile the model
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    # for tensorboard https://www.tensorflow.org/tensorboard/get_started
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train
    model.fit(x_train, y_train, epochs=5,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print('\nTest accuracy: {}'.format(test_acc))
    '''
    open terminal and print:
    
    tensorboard --logdir logs/fit
    '''
    # saving the model https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#save_your_model
    import tempfile

    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model:')

    print('done')
    return


if __name__ == '__main__':
    main()