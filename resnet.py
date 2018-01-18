"""Define a residual network that takes in calcium data and outputs spike
predictions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util.nncomponents import dense_batch_relu, \
summarize_layer, conv1d_batch_norm, scaled_xavier
from util.nnio import pad_to_batch_size

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow import summary

def residual_network(features, mode, config):
    seq_len = config['window_padding'] * 2 + config['window_size']
    batch_size = config['batch_size']
    input_layer = tf.reshape(features['X'], shape=(
        batch_size, seq_len, 1), name="initial_reshape")

    if config['global_features']:
        global_features = features['global_features']

        global_h1 = dense_batch_relu(global_features,
                                     mode == learn.ModeKeys.TRAIN,
                                     'global_h1',
                                     config['N_global_intermediate_variables'])

        global_h2 = dense_batch_relu(global_h1,
                                     mode == learn.ModeKeys.TRAIN,
                                     'global_h2',
                                     config['N_global_latent_variables'],
                                     use_relu=False)

        global_features_processed = global_h2

        global_h2 = tf.layers.dropout(global_h2,
                                      rate=config['dropout'],
                                      training=mode == learn.ModeKeys.TRAIN)

        global_h2 = tf.nn.softmax(global_h2)

    if config['global_features']:
        shape = (config['conv_window_size'],
                 config['N_latent_variables'],
                 config['N_global_latent_variables'])
        W1 = tf.get_variable('W1',
                             shape=(shape[0], 1, shape[1] * shape[2]),
                             initializer=tf.contrib.layers.variance_scaling_initializer(
                                 factor=.3,
                                 mode='FAN_AVG',
                                 uniform=False))

        # svd_weights = tf.matmul(global_h2, W1)
        b1 = tf.get_variable('b1', shape=(1, 1, shape[1] * shape[2]))
        h1 = tf.nn.conv1d(input_layer,
                          W1,
                          stride=1,
                          padding='VALID') + b1
        winsize = (config['window_size'] +
                     2 * config['window_padding'] -
                     config['conv_window_size'] + 1)

        h1 = tf.reshape(h1, [config['batch_size'],
                             winsize * config['N_latent_variables'],
                             config['N_global_latent_variables']])
        # global
        h1 = tf.einsum('aij,aj->ai', h1, global_h2)
        h1 = tf.reshape(h1, [config['batch_size'],
                             winsize,
                             config['N_latent_variables']], name='h1')
    else:
        h1 = tf.layers.conv1d(
            input_layer,
            config['N_latent_variables'],
            config['conv_window_size'],
            strides=1,
            use_bias=True,
            kernel_initializer=scaled_xavier(.005),
            bias_initializer=tf.constant_initializer(.01),
            padding='valid',
            name='h1',
            activation=None)

    summarize_layer('h1', h1)
    h1_norm = tf.layers.batch_normalization(
        h1,
        axis=2,
        training=(mode == learn.ModeKeys.TRAIN),
        name='h1_norm')

    h1_norm = tf.layers.dropout(h1_norm,
                                rate=config['dropout'],
                                training=mode == learn.ModeKeys.TRAIN)

    h1_out = tf.nn.relu(h1_norm, name='h1_out')
    summarize_layer('h1_out', h1_out)

    sum_layer = tf.identity(h1_out)

    for i in range(config['N_hidden_layers']):
        middle_layer = tf.layers.conv1d(
            sum_layer,
            config['N_latent_variables'],
            config['adj_window_size'],
            strides=1,
            bias_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False),
            kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False),
            padding='valid',
            activation=None)

        middle_layer_norm = tf.layers.batch_normalization(
            middle_layer,
            axis=2,
            training=(mode == learn.ModeKeys.TRAIN)
        )

        middle_layer_norm = tf.layers.dropout(middle_layer_norm,
                            rate=config['dropout'],
                            training=mode == learn.ModeKeys.TRAIN)

        middle_layer_out = tf.nn.relu(middle_layer_norm)
        summarize_layer('middle_layer%d_out' % i, middle_layer_out)

        dw = int((config['adj_window_size'] - 1) / 2)
        sum_layer = tf.slice(sum_layer,
                             [0,  dw, 0],
                             [-1, sum_layer.shape[1].value - 2 * dw, -1])

        sum_layer += middle_layer_out

    shape = (config['batch_size'], config['window_size'],
             config['N_latent_variables'])

    flattened_inputs = tf.reshape(sum_layer, [shape[0] * shape[1], shape[2]])

    output = tf.layers.dense(flattened_inputs,
                             units=1,
                             name='output_layer',
                             activation=None,
                             kernel_initializer=tf.random_normal_initializer(
                                 mean=0.0, stddev=0.0001),
                             bias_initializer=tf.constant_initializer(.2),
                             )
    output = tf.reshape(output, shape=[shape[0], shape[1]])

    if config['global_features'] == 'prepost':
        post_0 = dense_batch_relu(global_features_processed,
                                     mode == learn.ModeKeys.TRAIN,
                                     'global_post_0',
                                     config['N_global_latent_variables'],
                                     use_relu=True,
                                     weight_scale=.2)

        post = dense_batch_relu(post_0,
                         mode == learn.ModeKeys.TRAIN,
                         'global_post',
                         1,
                         use_relu=False,
                         weight_scale=.2)
        post = tf.log(1 + tf.exp(post))
        summarize_layer("post", post)
        output = output * post

    summarize_layer('cropped_output_before_norm', output)
    output = tf.layers.batch_normalization(output,
                                           name='output_normalized',
                                           training=(mode == learn.ModeKeys.TRAIN))
    output = tf.nn.relu(output)
    summarize_layer('cropped_output', output)
    output = tf.reshape(output, shape=(config['batch_size'], config['window_size']))

    neuron_ids = tf.identity(features['neuron_ids'])
    normalizers = tf.get_variable(
        'normalizers_0',
        shape=(config['N_neurons'] + 1,),
        initializer=tf.ones_initializer(dtype=tf.float32),
        trainable=True)

    tf.add_to_collection('normalizers_0', tf.GraphKeys.TRAINABLE_VARIABLES)

    normalizers = normalizers / (1 + tf.reduce_mean(normalizers))
    summarize_layer('normalizers', normalizers)
    summary.scalar("fraction_of_zeros_in_output", tf.nn.zero_fraction(output))
    outputs = {'output': output}
    return outputs, normalizers

def get_config():
    config = {'feature_type': 'continuous',
              'window_size': 64,
              'conv_window_size': 33,
              'adj_window_size': 9,
              'model_name': "conservative_model",
              'eval_every': 2000,
              'niter': int(170e3),
              'decay_every': int(110e3),
              'decay_multiplier': .1,
              'mean_obs_per': 3e4,
              'num_training_sets': 10,
              'batch_size': 128,
              'network_fn': residual_network,
              'alpha': 1e-4,
              'N_neurons': 174,
              'N_latent_variables': 32,
              'N_global_features': 8, # <= 32, can truncate the representation
              'N_global_latent_variables': 4, # the dimensionality of the adaptation
              'N_global_intermediate_variables' : 16, # hidden features in the mini-net that does adaptation
              'N_hidden_layers': 7,
              'validation_cycle': 1,
              'validation_bycell': 'sub',
              'use_normalizer': True,
              'boundary_conditions': 'mirror',
              'global_features': 'prepostunsupervised', #'prepost' uses predefined global features instead of the learned ones.
              'global_lengthscale': 5000, # for prepost
              'dropout': 0.3,
              'state_size': 1,
              'refine_recording': None,
              'early_stopping_rounds': int(1e5),
              }

    if config['refine_recording'] is not None:
        config['eval_every'] /= 5
        config['early_stopping_rounds'] = 1e4
        config['niter'] = 100000

    if config['conv_window_size'] % 2 == 0:
        raise NotImplementedError("Even conv_window_size")
    window_padding = int(
        config['N_hidden_layers'] * (config['adj_window_size'] - 1) / 2 + (config['conv_window_size'] - 1) / 2)
    config['window_padding_l'] = window_padding
    config['window_padding_r'] = window_padding
    config['window_padding'] = window_padding

    return config
