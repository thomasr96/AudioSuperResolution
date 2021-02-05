# Subpixel operation as used in https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers/convolution/super_resolution.py

import numpy as np
import tensorflow as tf

def phase_shift_tensorlayer(r):
    def _PS(I):
        X = tf.transpose(a=I, perm=[2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space(input=X, block_shape=[r], crops=[[0, 0]])  # (1, r*w, b)
        X = tf.transpose(a=X, perm=[2, 1, 0])
        return X
    return _PS
