import librosa
import numpy as np
import random
from pydub import AudioSegment
import os
import glob

def audiosegment_to_ndarray(audiosegment):
    samples = audiosegment.get_array_of_samples()
    samples_float = librosa.util.buf_to_float(samples,n_bytes=2,dtype=np.float32)
    if audiosegment.channels==2:
#         print('11')
        sample_left= np.copy(samples_float[::2])
        sample_right= np.copy(samples_float[1::2])
        sample_all = np.array([sample_left,sample_right])
    else:
#         print('22')
        sample_all = samples_float
        
        
    return [sample_all, audiosegment.frame_rate]

def vctk_tracklist_to_partition(tracklistfile, split):
    train_size = split[0]
    val_size = split[1]

    with open(tracklistfile, 'r') as txtfile:
        tracks = txtfile.readlines()
    for i, track in enumerate(tracks):
        try:
            tracks[i] = track.strip('\n')
        except:
#             print(track.strip())
            pass
#         tracks[i] = os.path.splitext(os.path.split(track)[1])[0]
#     print(tracks)
    random.shuffle(tracks)
    
    trainlen = int(len(tracks)*train_size)
    vallen = trainlen + int(len(tracks)*val_size)

    
    
    partition = {'train':[], 'validation':[]}
    levels = {}
#     for class_dir in glob.glob1(traindir, '*'):
    for i, track in enumerate(tracks):
        if i < trainlen:
            partition['train'].append(track)
        else:
            partition['validation'].append(track)
            
        levels[track] = int(random.randint(0,1))
        
            
    
    return partition, levels


def fma_tracklist_to_partition(tracklistfile, split):
    train_size = split[0]
    val_size = split[1]

    with open(tracklistfile, 'r') as txtfile:
        tracks = txtfile.readlines()
    for i, track in enumerate(tracks):
        try:
            tracks[i] = track.remove('\n')
        except:
            pass
        tracks[i] = os.path.splitext(os.path.split(track)[1])[0]
    
    random.shuffle(tracks)
    
    trainlen = int(len(tracks)*train_size)
    vallen = trainlen + int(len(tracks)*val_size)

    
    
    partition = {'train':[], 'validation':[]}
    levels = {}
#     for class_dir in glob.glob1(traindir, '*'):
    for i, track in enumerate(tracks):
        if i < trainlen:
            partition['train'].append(track)
        else:
            partition['validation'].append(track)
            
        levels[track] = int(random.randint(0,1))
        
            
    
    return partition, levels

def get_input_shape(trackslistfile, songlength, sr):
    with open(trackslistfile, 'r') as txt:
        tracks = txt.readlines()
    track = tracks[0].replace('\n', '')
    
    y1 = AudioSegment.from_mp3(track)                
    y1 = y1.set_channels(1)
    y1 = y1.set_frame_rate(sr)
    y1 = y1[:songlength]
    y1, sr = audiosegment_to_ndarray(y1)
    
    inputLength = y1.shape[0]
    input_shape=(inputLength, 1)

    return input_shape

def get_input_shape_2(track, songlength, sr):
    
    y1 = AudioSegment.from_mp3(track)                
    y1 = y1.set_channels(1)
    y1 = y1.set_frame_rate(sr)
    y1 = y1[:songlength]
    y1, sr = audiosegment_to_ndarray(y1)
    
    inputLength = y1.shape[0]
    input_shape=(inputLength, 1)

    return input_shape

def make_song_list_file(songlistfile):
    songlines = []
    for directory in glob.glob('/home/tretzlof/scratch/fma_full/*'):
        if os.path.isdir(directory):
    #         for songfile in glob.glob(os.path.join(directory, '*')):
            songlines += [filename+'\n' for filename in glob.glob(os.path.join(directory, '*'))]
    with open(songlistfile, 'w+') as file:
        file.writelines(songlines)

