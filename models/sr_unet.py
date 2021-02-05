# Creates UNet model class 

import sys
sys.path.insert(0,'./layers') 
sys.path.insert(0,'../helpers') 


import random
import os
import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer
import time
import datetime

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import importlib

import helper_functions
importlib.reload(helper_functions)
from helper_functions import get_input_shape, make_song_list_file, vctk_tracklist_to_partition

import pickle


class SRUNet():
    def __init__(self, input_shape, input_length, songlength, datapath, save_pickle_file_path, sr_hr, ratio, 
                 tracklistfile, training_generator, validation_generator, number_of_layers, lr, 
                 create_generator, create_content_extractor, batchsize, epochs, skip_connections, 
                 use_content_extractor, saved_weights_dir_path, content_weights_file_path, save_epoch_step, n_mels, nfft, fmin, fmax, 
                 power_melgram):
        
        self.input_shape = input_shape
        self.input_length = input_length
        self.datapath = datapath 
        self.save_pickle_file_path = save_pickle_file_path
        self.songlength = songlength 
        self.sr_hr = sr_hr 
        self.ratio = ratio 
        self.tracklistfile = tracklistfile 
        
        self.number_of_layers = number_of_layers
        self.lr = lr
        self.skip_connections = skip_connections
        self.batchsize = batchsize 
        self.epochs = epochs
        
        
        self.n_mels = n_mels
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax 
        self.power_melgram = power_melgram
        self.hoplength = int(self.sr_hr/1000*10)
        self.sr_lr = int(self.sr_hr/self.ratio)
        
        
        
        
        self.data_output_generator, self.train_step_num = self.create_data_generator(training_generator)
                
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mean_squared_error = MeanSquaredError()

        self.generator_optimizer = Adam(self.lr)
        
        generator_input, generator_output = create_generator(self.input_shape, self.number_of_layers, 
                                                               self.ratio, self.skip_connections)
        
        self.generator = Model(inputs=generator_input, outputs=generator_output)
        
        if use_content_extractor:
            self.content_weights_file_path = content_weights_file_path

            content_extractor_input, content_extractor_output = create_content_extractor(self.input_length, self.sr_hr, 
                                                                                         self.n_mels, self.hoplength, 
                                                                                         self.nfft, self.fmin, 
                                                                                         self.fmax, self.power_melgram)

            
            self.content_extractor = Model(inputs=content_extractor_input, outputs=content_extractor_output)

            self.content_extractor.load_weights(content_weights_file_path)

            
            self.train_step = self.train_step_content
            self.val_step = self.val_step_content

        else:
            self.train_step = self.train_step_no_content
            self.val_step = self.val_step_no_content
            
        self.saved_weights_dir_path = saved_weights_dir_path
        
        self.save_epoch_step = save_epoch_step
        
    def create_data_generator(self, training_generator):       

        prefetch_batch_buffer = 1000
        training_dataset = tf.data.Dataset.from_generator(training_generator.data_generation, 
                                                          output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape((self.input_length, 1)), 
                                                                tf.TensorShape((self.input_length, 1))))
        training_dataset = training_dataset.batch(64)
        training_dataset = training_dataset.prefetch(prefetch_batch_buffer)

        
        return iter(training_dataset), len(training_generator)



    def generator_loss_content(self, hr_output, generated_output, hr_content_output, generated_content_output):
        lambda_f = .5
        
        L_2 = self.mean_squared_error(hr_output, generated_output)
        L_content = self.mean_squared_error(hr_content_output, generated_content_output)
        
        return L_2 + lambda_f * L_content 
    
    def generator_loss_no_content(self, hr_output, generated_output):
        L_2 = self.mean_squared_error(hr_output, generated_output) 
        return L_2 

    @tf.function
    def train_step_content(self, lr_input, hr_output):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(lr_input)
            hr_content_output = self.content_extractor(hr_output)
            generated_content_output = self.content_extractor(generated_output)
            gen_loss = self.generator_loss_content(hr_output, generated_output, hr_content_output, generated_content_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss
    @tf.function
    def val_step_content(self, lr_input, hr_output):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(lr_input)
            hr_content_output = self.content_extractor(hr_output)
            generated_content_output = self.content_extractor(generated_output)
    

            gen_loss = self.generator_loss_content(hr_output, generated_output, hr_content_output, generated_content_output)


        return gen_loss
    
    @tf.function
    def train_step_no_content(self, lr_input, hr_output):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(lr_input)
    
            gen_loss = self.generator_loss_no_content(hr_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss
    @tf.function
    def val_step_no_content(self, lr_input, hr_output):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(lr_input)
   

            gen_loss = self.generator_loss_no_content(hr_output, generated_output)

       
        return gen_loss
    
    def train(self):
        avg_losses = []
        for epoch in range(self.epochs):
            gen_loss = []
            
            
            start = time.time()
            dashes = ''.join(['-' for i in range(20)])
            print('\n\n{}Epoch: {}{}'.format(dashes, epoch, dashes))
            
            for train_step in range(self.train_step_num):
                train_step_time = time.time()
                lr_input, hr_output = self.data_output_generator.get_next()
                lr_input = lr_input.numpy()
                hr_output = hr_output.numpy()
                gen_loss.append(self.train_step(lr_input, hr_output))
                
                print('Train step {}/{}, Generator loss: {:.6}'
                      .format(train_step, self.train_step_num, gen_loss[-1] ))
                
            if (epoch + 1) % self.save_epoch_step == 0:

                self.generator.save_weights(os.path.join(self.saved_weights_dir_path, 
                                                         'generator_weights_{}_'.format(epoch)), save_format='tf')
    
            avg_losses.append(sum(gen_loss)/len(gen_loss))
            
            with open(self.save_pickle_file_path, 'wb+') as f:
                pickle.dump(avg_losses, f)
                        
            print ('Time for epoch {} is {} sec\n avg generator loss: {:.6}'
                   .format(epoch + 1, time.time()-start, avg_losses[-1], ))