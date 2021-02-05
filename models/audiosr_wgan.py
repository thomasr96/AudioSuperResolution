# Creates WGAN model class 

import sys  
sys.path.insert(0,'./layers') 
sys.path.insert(0,'../helpers') 

import random
import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import importlib


import audio_autoencoder
importlib.reload(audio_autoencoder)
from audio_autoencoder import create_autoencoder

import vggish
importlib.reload(vggish)
from vggish import create_VGGish

import helper_functions
importlib.reload(helper_functions)
from helper_functions import get_input_shape, make_song_list_file, vctk_tracklist_to_partition

import ps_discriminator
importlib.reload(ps_discriminator)
from ps_discriminator import create_discriminator
import pickle

class AudioASRWGAN():
    def __init__(self, input_shape, input_length, songlength, datapath, save_pickle_file_path, sr_hr, ratio, tracklistfile, 
                 training_generator, validation_generator, number_of_layers, lr, create_discriminator, 
                 create_autoencoder, create_content_extractor, batchsize, epochs, skip_connections,
                 use_content_extractor, saved_weights_dir_path, content_weights_file_path, n_critic, clip_value, 
                 save_epoch_step, n_mels, nfft, fmin, fmax, power_melgram):
        self.use_content_extractor = use_content_extractor
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
#         self.num_workers = num_workers 
        self.content_weights_file_path = content_weights_file_path
        self.n_critic = n_critic 
        self.clip_value = clip_value

        self.n_mels = n_mels
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax 
        self.power_melgram = power_melgram
        self.hoplength = int(sr_hr/1000*10)
        self.sr_lr = int(sr_hr/ratio)

        self.data_output_generator, self.train_step_num = self.create_data_generator(training_generator)

        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mean_squared_error = MeanSquaredError()

        self.critic_loss = self.wasserstein_loss
        self.generator_loss = self.generator_loss_content_extractor

        self.generator_optimizer = Adam(lr)
        self.critic_optimizer = Adam(lr)

        generator_input, generator_output = create_autoencoder(input_shape, number_of_layers, ratio, skip_connections)

        critic_input, critic_output = create_discriminator(input_shape)

        if use_content_extractor:
            content_extractor_input, content_extractor_output = create_content_extractor(self.input_length, 
                                                                                         self.sr_hr, self.n_mels, 
                                                                                         self.hoplength, self.nfft, 
                                                                                         self.fmin, self.fmax, 
                                                                                         self.power_melgram)
            self.content_extractor = Model(inputs=content_extractor_input, outputs=content_extractor_output)

            self.content_extractor.load_weights(content_weights_file_path)
            self.train_generator_step = self.train_generator_step_content_extractor
            self.val_step = self.val_generator_step_content_extractor
        else:
                                             
            self.train_generator_step = self.train_generator_step_no_content_exractor
            self.val_step = self.val_generator_step_no_content_exractor


        self.generator = Model(inputs=generator_input, outputs=generator_output)
        self.critic = Model(inputs=critic_input, outputs=critic_output)
        
        self.saved_weights_dir_path = saved_weights_dir_path

        #self.self. checkpoint_prefix = os.path.join(checkpoint_dir_path, "ckpt")
        #self.self. checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
        #self.self.self.self.self.self.self.self.self.self.  critic_optimizer=critic_optimizer,
        #self.self.self.self.self.self.self.self.self.self.  generator=generator, critic=critic)

        self.save_epoch_step = save_epoch_step
        
        

    def create_data_generator(self, training_generator):       
#         enqueuer = OrderedEnqueuer(training_generator, use_multiprocessing=True, shuffle=True)
#         enqueuer.start(workers=num_workers, max_queue_size=10)
        
# validation_generator = VCTKGenerator(partition['validation'], labels, songlength, batch_size=batchsize, 
#                                      input_shape=input_shape, datapath=datapath, ratio=ratio, sr_lr=sr_lr, sr_hr=sr_hr)


        prefetch_batch_buffer = 1000
        training_dataset = tf.data.Dataset.from_generator(training_generator.data_generation, 
                                                          output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape((self.input_length, 1)), 
                                                                tf.TensorShape((self.input_length, 1))))
        training_dataset = training_dataset.batch(64)
        training_dataset = training_dataset.prefetch(prefetch_batch_buffer)
#         self.training_dataset = iter(training_dataset)
        
        return iter(training_dataset), len(training_generator)

    def wasserstein_loss(self, y_true, y_pred):
            return K.mean(y_true * y_pred)

    def generator_loss_content_extractor(self, hr_output, generated_output, hr_content_output, 
                                         generated_content_output, criticized_generated_output):
        lambda_f = 1.
        lambda_adv = .001
        L_2 = self.mean_squared_error(hr_output, generated_output)
        L_content = self.mean_squared_error(hr_content_output, generated_content_output)
        L_adv = self.cross_entropy(tf.ones_like(criticized_generated_output), criticized_generated_output)
        return L_2 + lambda_f * L_content + lambda_adv*L_adv
    
    def generator_no_content_extractor_loss(self, hr_output, generated_output, criticized_generated_output):
        lambda_adv = .001
        L_2 = self.mean_squared_error(hr_output, generated_output)
        L_adv = self.cross_entropy(tf.ones_like(criticized_generated_output), criticized_generated_output)
        return L_2 + lambda_adv*L_adv


    @tf.function
    def train_critic_step(self, lr_input, hr_output):
        with tf.GradientTape() as critic_tape:
         # Generate a batch of new images
            generated_output = self.generator(lr_input)
            criticized_hr_output  = self.critic(hr_output)
            criticized_generated_output = self.critic(generated_output)

            critic_loss_value = self.critic_loss(criticized_hr_output, criticized_generated_output)

        gradients_of_critic = critic_tape.gradient(critic_loss_value, self.critic.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, 
                                                         self.critic.trainable_variables))
        return critic_loss_value
    
    @tf.function
    def train_generator_step_content_extractor(self, lr_input, hr_output):
        with tf.GradientTape() as gen_tape:
            generated_output = self.generator(lr_input)

            criticized_generated_output = self.critic(generated_output)

            hr_content_output = self.content_extractor(hr_output)
            generated_content_output = self.content_extractor(generated_output)
    #         print(hr_output, generated_output, hr_content_output, generated_content_output, 
    #                    criticized_generated_output)

            gen_loss = self.generator_loss_content_extractor(hr_output, generated_output, hr_content_output, 
                                                        generated_content_output, criticized_generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)


        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    @tf.function
    def val_generator_step_content_extractor(self, lr_input, hr_output):
        with tf.GradientTape() as gen_tape:
            generated_output = self.generator(lr_input)

            criticized_generated_output = self.critic(generated_output)

            hr_content_output = self.content_extractor(hr_output)
            generated_content_output = self.content_extractor(generated_output)
    #         print(hr_output, generated_output, hr_content_output, generated_content_output, 
    #                    criticized_generated_output)

            gen_loss = self.generator_loss_content_extractor(hr_output, generated_output, hr_content_output, 
                                                        generated_content_output, criticized_generated_output)

        return gen_loss

    @tf.function
    def train_generator_step_no_content_exractor(self, lr_input, hr_output):
        with tf.GradientTape() as gen_tape:
            generated_output = self.generator(lr_input)

            criticized_generated_output = self.critic(generated_output)

            gen_loss = self.generator_no_content_extractor_loss(hr_output, generated_output, 
                                                                criticized_generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)


        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss
    
    @tf.function
    def val_generator_step_no_content_exractor(self, lr_input, hr_output):
        with tf.GradientTape() as gen_tape:
            generated_output = self.generator(lr_input)

            criticized_generated_output = self.critic(generated_output)

            gen_loss = self.generator_no_content_extractor_loss(hr_output, generated_output, 
                                                                criticized_generated_output)        

        return gen_loss

    def train(self):
        avg_losses = {'avg_critic_loss' : [], 'avg_gen_loss' : []}
        
        for epoch in range(self.epochs):
            gen_loss = []
            critic_loss = []

            start = time.time()
            dashes = ''.join(['-' for i in range(20)])
            print('\n\n{}Epoch: {}{}'.format(dashes, epoch, dashes))
            for train_step in range(self.train_step_num):
                train_step_time = time.time()
                lr_input, hr_output = [], []
                for iii in range(self.n_critic):
                    lr_input, hr_output = next(self.data_output_generator)

                    critic_loss.append(self.train_critic_step(lr_input, hr_output))

                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                gen_loss.append(self.train_generator_step(lr_input, hr_output))

#                 if(train_step % 10 == 0):
                    
                print('Train step {}/{} \n Gen loss: {:.6}, Critic loss: {:.6}, time: {:.6}s'
                      .format(train_step, self.train_step_num, gen_loss[-1], critic_loss[-1], 
                              time.time()-train_step_time))
#                 break
            if (epoch + 1) % self.save_epoch_step == 0:
    #                 checkpoint.save(file_prefix = checkpoint_prefix)
                self.generator.save_weights(os.path.join(self.saved_weights_dir_path, 
                                                         'generator_weights_{}_'.format(epoch)), save_format='tf')
        
#                 self.generator.save_weights(os.path.join(self.saved_weights_dir_path, 
#                                                          'generator_weights_{}_.hdf5'.format(epoch)))
                self.critic.save_weights(os.path.join(self.saved_weights_dir_path, 
                                                             'critic_weights_{}_'.format(epoch)), save_format='tf')
            
            avg_losses['avg_critic_loss'].append(sum(critic_loss)/len(critic_loss))
            avg_losses['avg_gen_loss'].append(sum(gen_loss)/len(gen_loss))
            
            with open(self.save_pickle_file_path, 'wb+') as f:
                pickle.dump(avg_losses, f)
                
            
            print ('Time for epoch {} is {} sec\n avg gen loss: {:.6}, avg critic loss: {:.6}'
                   .format(epoch + 1, time.time()-start, avg_losses['avg_critic_loss'][-1], avg_losses['avg_gen_loss'][-1]))
