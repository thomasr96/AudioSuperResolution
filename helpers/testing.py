import sys  
sys.path.insert(0,'/home/tretzlof/projects/def-pft3/tretzlof/audio_super_resolution/src/generators') 
sys.path.insert(0,'/home/tretzlof/projects/def-pft3/tretzlof/audio_super_resolution/src/models') 
sys.path.insert(0,'/home/tretzlof/projects/def-pft3/tretzlof/audio_super_resolution/src/helpers') 

import pickle

import random
import os
import numpy as np
import time
import glob
from pydub import AudioSegment

import librosa
import numpy as np
from scipy.signal import decimate
from scipy import interpolate

import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import importlib

import vctk_audio_generator_2
importlib.reload(vctk_audio_generator_2)
from vctk_audio_generator_2 import VCTKGenerator

import audio_autoencoder
importlib.reload(audio_autoencoder)
from audio_autoencoder import create_autoencoder

import vggish
importlib.reload(vggish)
from vggish import create_VGGish

import helper_functions
importlib.reload(helper_functions)
from helper_functions import get_input_shape, make_song_list_file, vctk_tracklist_to_partition, audiosegment_to_ndarray

import ps_discriminator
importlib.reload(ps_discriminator)
from ps_discriminator import create_discriminator

import sr_unet
importlib.reload(sr_unet)
from sr_unet import SRUNet

import audiosr_gan
importlib.reload(audiosr_gan)
from audiosr_gan import AudioSRGan

import audiosr_wgan
importlib.reload(audiosr_wgan)
from audiosr_wgan import AudioASRWGAN


def get_segments_from_file(filename, songlength):
    audio = AudioSegment.from_file(filename)
    
    # segments = np.array([0 for ])
    segments = []
    for j in range(0, len(audio), songlength):
        segment = []
        if j + songlength < len(audio):
            segment = audio[j:j+songlength]
        else:
            continue
    
        segments += [segment]
        
    return segments

def get_batch_from_audio(audio_list, batch_size, dim, ratio, sr_lr, sr_hr):
        
        X = np.empty((batch_size, dim,1))
#         X = np.empty((self.batch_size, 1, self.dim))
        y = np.empty((batch_size, dim, 1))
#         y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
#         might also think about generating 64 samples from 4 songs
        randomfile = ''
    
#         ids = [list_IDs_temp[random.randint(0, len(list_IDs_temp)-1)] for jj in range(4)]

        for i, y_track in enumerate(audio_list):
#             print(i)
            y_track = y_track.set_channels(1)

#             y_lr = y_track.set_frame_rate(sr_hr)
            y_hr = y_track.set_frame_rate(sr_hr)

#             y_lr, sr = audiosegment_to_ndarray(y_lr)
            y_hr, sr = audiosegment_to_ndarray(y_hr)
            
            y_lr = decimate(y_hr, ratio)
            
            y_lr = upsample(y_lr, ratio)
            
            assert len(y_hr) % ratio == 0
            assert len(y_lr) == len(y_hr)
            
            avg_length = 20



            
            if len(y_lr) < dim:
                print(2)
                for ii in range(dim-len(y_lr)):

                    y_lr = np.append(y_lr, [sum(y_lr[len(y_lr)-avg_length:])/avg_length])

            X[i,] = np.reshape(y_lr,(y_lr.shape[0], 1))

            y[i,] = np.reshape(y_hr,(y_hr.shape[0], 1))
            
        return X, y
    
def upsample(y_lr, r):
    y_lr = y_lr.flatten()
    y_hr_len = len(y_lr) * r
    y_sp = np.zeros(y_hr_len)

    i_lr = np.arange(y_hr_len, step=r)
    i_hr = np.arange(y_hr_len)

    f = interpolate.splrep(i_lr, y_lr)

    y_sp = interpolate.splev(i_hr, f)

    return y_sp

def get_vctk_dir_name_from_data_file(file_name):
    return os.path.splitext(os.path.split(file_name.strip())[1])[0].split('_')[0][1:]

def reformat_model_arrays(array_tuple):
    new_arrays = [[] for i in range(len(array_tuple))]
    
    for i, array_list in enumerate(array_tuple):
        for array in array_list:
#             print(array.reshape(len(array)))
            new_arrays[i] += [array.reshape(len(array))]
    return new_arrays

def librosa_array_to_wav(y_batch, file_name, sr):
    y_song = np.array([])
    for y in y_batch:
    #     y = y.reshape(len(y))
        y_song = np.append(y_song,  y)    
    librosa.output.write_wav(file_name, y_song.astype('f'), sr)
    
def librosa_array_to_pydub(y_batch, file_name, sr):
    librosa_array_to_wav(y_batch, file_name, sr)
    return AudioSegment.from_file(file_name)


def make_val_list(file_path_list, songlength, batch_size):
    batch = []
    
    for file_path in file_path_list:
        batch += get_segments_from_file(file_path, songlength)
        if len(batch) >= batch_size:
            return batch[:batch_size]
    print('Not enough files for batch size {}'.format(batch_size))
    return(batch)


def test_on_audio_file(model, filename, songlength, input_shape, ratio, sr_lr, sr_hr):
    audio_list = get_segments_from_file(filename, songlength)
    dim = input_shape[0]
    X_batch, y_batch = get_batch_from_audio(audio_list, len(audio_list), dim, ratio, sr_lr, sr_hr)
    y_pred = model.predict_on_batch(X_batch).numpy()
    X_batch, y_batch, y_pred = reformat_model_arrays((X_batch, y_batch, y_pred))
    
    test_file = 'test.wav'
    lr_audio = librosa_array_to_pydub(X_batch, test_file, sr_hr)
    hr_audio = librosa_array_to_pydub(y_batch, test_file, sr_hr)
    generated_audio = librosa_array_to_pydub(y_pred, test_file, sr_hr)
    
    return lr_audio, hr_audio, generated_audio