# original VGGish model https://github.com/tensorflow/models/tree/master/research/audioset/vggish
# Get weights from https://drive.google.com/file/d/1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6/view

import sys
sys.path.insert(0, './layers')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, add, LeakyReLU, Flatten
from tensorflow.keras.layers import Activation, Dropout, Conv1D, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.optimizers import Adam

from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

def create_VGGish(input_length, sr_hr, n_mels, hoplength, nfft, fmin, fmax, power_melgram, pooling='avg'):
    
    
    X = Input(shape = (input_length, 1), name='input_1')
    
    x = X
    
    x = Reshape((1, input_length))(x)

    x = Spectrogram(n_dft=nfft, n_hop=hoplength,
                             padding='same',
                             return_decibel_spectrogram=True,
                             trainable_kernel=False,
                             name='stft')(x)
    
    x = Normalization2D(str_axis='freq')(x)
    
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)


    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    return X, x


def add_mel_to_VGGish(content_weights_file_path_og, input_length, sr_hr, n_mels, hoplength, nfft, fmin, fmax, power_melgram):
    
    
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.
    pooling='avg'
    X = Input(shape=(NUM_FRAMES,NUM_BANDS, 1), name='nob')
    x = X
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)


    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    model = Model(inputs=X, outputs=x)
    
    model.load_weights(content_weights_file_path_og)
    
    X = Input(shape = (1, input_length), name='input_1')
    x = X
    x = Spectrogram(n_dft=nfft, n_hop=hoplength,
                             padding='same',
                             return_decibel_spectrogram=True,
                             trainable_kernel=False,
                             name='stft')(x)

    x = Normalization2D(str_axis='freq')(x)
    
    no_input_layers = model.layers[1:]
    
    for layer in no_input_layers:
        x = layer(x)
    return Model(inputs=X, outputs=x)
