# Creates a data generator to train on

import os
import random
import math
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.signal import decimate
from scipy import interpolate

class VCTKGenerator():

    def __init__(self, list_IDs, songlength, batch_size, input_shape, datapath, ratio, sr_lr, sr_hr, n_channels=1, 
                 shuffle=True):
        
#         self.labels = labels
        self.list_IDs = list_IDs
        self.songlength = songlength
        self.ratio = ratio
        self.datapath = datapath
        self.dim = input_shape[0]
        self.sr_lr = sr_lr
        self.sr_hr = sr_hr
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def audiosegment_to_ndarray(self, audiosegment):
        samples = audiosegment.get_array_of_samples()
        samples_float = librosa.util.buf_to_float(samples,n_bytes=2,
                                          dtype=np.float32)
        if audiosegment.channels==2:
            sample_left= np.copy(samples_float[::2])
            sample_right= np.copy(samples_float[1::2])
            sample_all = np.array([sample_left,sample_right])
        else:
            sample_all = samples_float


        return [sample_all, audiosegment.frame_rate]
    def upsample(self, y_lr, r):
        y_lr = y_lr.flatten()
        y_hr_len = len(y_lr) * r
        y_sp = np.zeros(y_hr_len)

        i_lr = np.arange(y_hr_len, step=r)
        i_hr = np.arange(y_hr_len)

        f = interpolate.splrep(i_lr, y_lr)

        y_sp = interpolate.splev(i_hr, f)

        return y_sp


    def data_generation(self):
        i = 0 
        while True:
            i += 1
#     hr examples are generated
            ID = random.choice(self.list_IDs)
            y_track = AudioSegment.from_file(ID)
            y_track = y_track[:self.songlength]
            y_track = y_track.set_channels(1)
            y_hr = y_track.set_frame_rate(self.sr_hr)
# librosa format, AudioSegment was originally used because librosa was too slow in loading files
            y_hr, sr = self.audiosegment_to_ndarray(y_hr)
# low res examples are generated from the hig res version
            y_lr = decimate(y_hr, self.ratio)

            y_lr = self.upsample(y_lr, self.ratio)

            assert len(y_hr) % self.ratio == 0
            assert len(y_lr) == len(y_hr)

            avg_length = 20




            if len(y_lr) < self.dim:
                for ii in range(self.dim-len(y_lr)):

                    y_lr = np.append(y_lr, [sum(y_lr[len(y_lr)-avg_length:])/avg_length])


            X = np.reshape(y_lr.astype('float32'),(y_lr.shape[0], 1))

            y = np.reshape(y_hr.astype('float32'),(y_hr.shape[0], 1))
            
            yield (X, y)