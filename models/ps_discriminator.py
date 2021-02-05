# Discriminator and critic used in GAN and WGAN models
# Adapted from  from https://github.com/chrisdonahue/wavegan

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, add, LeakyReLU, Flatten
from tensorflow.keras.layers import Activation, Dropout, Conv1D, Reshape, BatchNormalization, LeakyReLU

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from utils import Normalization2D

def apply_phaseshuffle(x, rad=2, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

def create_discriminator(input_shape, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=1):
    
#     batch_size = tf.shape(x)[0]
#     slice_len = int(x.get_shape()[1])
    
    X = Input(input_shape)
    
    x = X
    if use_batchnorm:
        batchnorm = lambda x: BatchNormalization(x)
    else:
        batchnorm = lambda x: x

    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x

    # Layer 0
    # [16384, 1] -> [4096, 64]
#     output = x
    
    x = (Conv1D(filters=dim, kernel_size=kernel_len, strides=4,  padding='SAME', name='downconv_0'))(x)
    x = LeakyReLU()(x)
    x = phaseshuffle(x)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    
    x = (Conv1D(filters=dim*2, kernel_size=kernel_len, strides=4,  padding='SAME', name='downconv_1'))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = phaseshuffle(x)

    # Layer 2
    # [1024, 128] -> [256, 256]
    x = (Conv1D(filters=dim*4,  kernel_size=kernel_len, strides=4,  padding='SAME', name='downconv_2'))(x)    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = phaseshuffle(x)

    # Layer 3
    # [256, 256] -> [64, 512]
    x = (Conv1D(filters=dim*8, kernel_size=kernel_len, strides=4,  padding='SAME', name='downconv_3'))(x)    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = phaseshuffle(x)
    
    # Layer 4
    x = (Conv1D(filters=dim*16, kernel_size=kernel_len, strides=4,  padding='SAME', name='downconv_4'))(x)    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = phaseshuffle(x)

    x = Flatten()(x)    
    x = (Dense(1, activation='sigmoid', name='output'))(x)
    

    return X, x