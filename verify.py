# -*- coding: utf-8 -*-
"""https://www.tensorflow.org/install/docker
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


def main():
    # print(tf.reduce_sum(tf.random.normal([1000, 1000])))
    print(tf.__version__)
    print(tf.config.experimental.list_physical_devices())
    print('ok')
    return


if __name__ == '__main__':
    main()