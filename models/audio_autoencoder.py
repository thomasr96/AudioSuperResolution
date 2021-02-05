# Creates UNet model
# Adapted from https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, add, Activation, Dropout, Conv1D, BatchNormalization, LeakyReLU
from keras.initializers import normal, orthogonal
import sys
sys.path.insert(0,'./models/layers') 
from subpixel import phase_shift_tensorlayer

# uses skip connections
def create_autoencoder_skip_connections(input_shape, number_of_layers, ratio):
    

    X = Input(shape=input_shape)


    x = X
    L = number_of_layers

    n_filters = [min(2**(6+i), 512) for i in range(1, number_of_layers)] + [min(2**(7+(number_of_layers-i+1)), 512) for i in range(1, number_of_layers+1)]
    n_filtersizes = [max(2**(7-i)+1,9) for i in range(1, number_of_layers)] + [max(2**(7-(number_of_layers-i+1))+1,9) for i in range(1, number_of_layers+1)]

    downsampling_l = []

    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):

            x = (Conv1D(filters=nf, kernel_size=fs, 
                  activation=None, padding='same', kernel_initializer=orthogonal(),
                  strides=2))(x)

            x = LeakyReLU(0.2)(x)

            downsampling_l.append(x)

          # bottleneck layer
  
    x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], 
          activation=None, padding='same', kernel_initializer=orthogonal(),
          strides=2))(x)
    x = Dropout(rate = .5)(x)

    x = LeakyReLU(0.2)(x)

    # upsampling layers
    for l, nf, fs, l_in in list(zip(*[range(L), n_filters, n_filtersizes, downsampling_l]))[::-1] :
    

        x = (Conv1D(filters=2*nf, kernel_size=fs, 
              activation=None, padding='same', kernel_initializer=orthogonal()))(x)
       
        x = Dropout(rate = .5)(x)
        x = Activation('relu')(x)
        
        x = Lambda(phase_shift_tensorlayer(2))(x)
        
        x = concatenate([x, l_in], axis=-1) 

    # final conv layer
    
    x = Conv1D(filters=2, kernel_size=9, 
            activation=None, padding='same', kernel_initializer='random_normal')(x)    

    x = Lambda(phase_shift_tensorlayer(2))(x)


    g = add([x, X])

    return X, g

# doesn't use skip connections
def create_autoencoder_no_skip_connections(input_shape, number_of_layers, ratio):
    

    X = Input(shape=input_shape)


    x = X
    L = number_of_layers

    n_filters = [min(2**(6+i), 512) for i in range(1, number_of_layers)] + [min(2**(7+(number_of_layers-i+1)), 512) for i in range(1, number_of_layers+1)]
    n_filtersizes = [max(2**(7-i)+1,9) for i in range(1, number_of_layers)] + [max(2**(7-(number_of_layers-i+1))+1,9) for i in range(1, number_of_layers+1)]

    downsampling_l = []

    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):


        x = (Conv1D(filters=nf, kernel_size=fs, 
              activation=None, padding='same', kernel_initializer=orthogonal(),
              strides=2))(x)
        x = LeakyReLU(0.2)(x)
        downsampling_l.append(x)

          # bottleneck layer
  
    x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], 
          activation=None, padding='same', kernel_initializer=orthogonal(),
          strides=2))(x)
    x = Dropout(rate = .5)(x)

    x = LeakyReLU(0.2)(x)

    # upsampling layers
    for l, nf, fs, l_in in list(zip(*[range(L), n_filters, n_filtersizes, downsampling_l]))[::-1] :
    

        x = (Conv1D(filters=2*nf, kernel_size=fs, 
              activation=None, padding='same', kernel_initializer=orthogonal()))(x)
       
        x = Dropout(rate = .5)(x)
        x = Activation('relu')(x)
        
        x = Lambda(phase_shift_tensorlayer(2))(x)
        
    # final conv layer
    
    x = Conv1D(filters=2, kernel_size=9, 
            activation=None, padding='same', kernel_initializer='random_normal')(x)    

    x = Lambda(phase_shift_tensorlayer(2))(x)

    return X, g

def create_autoencoder(input_shape, number_of_layers, ratio, skip_connections):
    if skip_connections:
        return create_autoencoder_skip_connections(input_shape, number_of_layers, ratio)
    else:
        return create_autoencoder_no_skip_connections(input_shape, number_of_layers, ratio)
