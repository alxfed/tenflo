# -*- coding: utf-8 -*-
"""https://www.tensorflow.org/guide/data
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)


def main():
    '''
    To create an input pipeline, you must start with a data source.
    For example, to construct a Dataset from data in memory,
    you can use
    tf.data.Dataset.from_tensors() or
    tf.data.Dataset.from_tensor_slices().
    Alternatively, if your input data is stored in a file in the
    recommended TFRecord format, you can use tf.data.TFRecordDataset().
    '''
    return


if __name__ == '__main__':
    main()
    print('\ndone')