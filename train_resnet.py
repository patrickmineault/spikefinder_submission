"""Fit a TF model for spike detection.

Fits a resnet with an initial adaptive layer. See the *.sh files in this
directory to see how to call this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import functools
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.layers.python.layers import optimizers as optimizers_lib
from tensorflow import summary

from util.nnio import eval_and_save, load_all, split_data, pad_to_batch_size
from util.nncomponents import dense_batch_relu, summarize_layer
import resnet

config = resnet.get_config()

tf.logging.set_verbosity(tf.logging.INFO)

def simple_model_fn(features,
                    labels,
                    mode,
                    params):
    """Model function for LN model."""
    features['alphas'] = tf.reshape(features['alphas'], (params['batch_size'], 1))
    features['neuron_ids'] = tf.reshape(features['neuron_ids'], (params['batch_size'], params['window_size']))
    features['w'] = tf.reshape(features['w'], (params['batch_size'], params['window_size']))
    features['global_features'] = tf.reshape(features['global_features'],
                                             (params['batch_size'], params['N_global_features']))
    features['X'] = tf.reshape(features['X'], (params['batch_size'],
                                               params['window_size'] + 2 * params['window_padding']))

    outputs, normalizers = params['network_fn'](features, mode, params)

    if mode == learn.ModeKeys.TRAIN:
        summarize_layer('labels', labels)

    output = outputs['output']
    summarize_layer('output', output)

    if params['use_normalizer']:
        multipliers = tf.gather(normalizers, features['neuron_ids'])
        output_norm = multipliers * output
    else:
        output_norm = tf.identity(output)

    output_norm_pre = output_norm
    output_norm = output_norm * features['alphas']

    loss = None
    train_op = None

    zero_fraction = tf.identity(tf.reduce_mean(
        tf.nn.zero_fraction(output_norm), keep_dims=True),
                                name="zero_fraction_tracker")

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=tf.reshape(labels, shape=(-1, )),
            predictions=tf.reshape(output_norm, shape=(-1, )),
            weights=tf.reshape(features['w'], shape=(-1, ))
        )

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        global_step = tf.contrib.framework.get_global_step()
        learning_rate = tf.train.exponential_decay(
            params['alpha'],
            global_step,
            params['decay_every'],
            params['decay_multiplier'],
            staircase=True,
            name='learning_rate')
        summary.scalar('learning_rate', learning_rate)
        summary.scalar('sum_weights_train', tf.reduce_sum(labels[:, 1]))
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer="Adam",
            clip_gradients=optimizers_lib.adaptive_clipping_fn())
    elif mode == learn.ModeKeys.EVAL:
        summary.scalar('sum_weights_eval', tf.reduce_sum(labels[:, 1]))

    # Generate Predictions
    predictions = {
        "relu_output": tf.identity(output_norm, name='relu_output'),
        "relu_coarse": tf.cast(tf.round(tf.reshape(output_norm, (-1, )) * 1e4),
                               tf.int32,
                               name='relu_coarse'),
    }

    if mode != learn.ModeKeys.INFER:
        eval_metric_ops = {
            "mse":
                tf.metrics.mean_squared_error(
                    labels=tf.reshape(labels, (-1, )),
                    predictions=tf.reshape(output_norm, (-1, )),
                    weights=tf.reshape(features['w'], shape=(-1, ))
                ),
        }
    else:
        eval_metric_ops = None

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

# Test this function
#config['batch_size'] = 25
#data = {}

def restrict_to_recording(all_data, all_labels, all_dataset_map, recording):
    recording_num, _, _ = zip(*all_dataset_map.values())
    cell_num = all_dataset_map.keys()
    neuron_ids = [i for i, x in zip(cell_num, recording_num) if x == recording]
    good_idx = np.in1d(all_data['neuron_ids'][:, 0], neuron_ids)
    all_data = {x: y[good_idx] for x, y in all_data.items()}
    if all_labels is not None:
        all_labels = all_labels[good_idx]
    all_dataset_map = {x: y for x, y in all_dataset_map.items() if x in neuron_ids}
    return all_data, all_labels, all_dataset_map

def get_relevant_data(dataset, config):
    all_data, all_labels, all_dataset_map = load_all(dataset,
                                                     config['num_training_sets'],
                                                     config)
    if config['refine_recording'] is not None:
        # Select only one recording.
        all_data, all_labels, all_dataset_map = restrict_to_recording(
            all_data,
            all_labels,
            all_dataset_map,
            config['refine_recording'])

    return all_data, all_labels, all_dataset_map

def get_spike_classifier(config):
    spike_classifier = learn.Estimator(
        model_fn=simple_model_fn,
        model_dir=config['model_name'],
        config=learn.RunConfig(save_checkpoints_secs=60,
                               log_device_placement=True,
                               keep_checkpoint_every_n_hours=1,
                               keep_checkpoint_max=int(1e6)),
        params=config)
    return spike_classifier

def main(unused_argv):
    # Load training and eval data
    tf.logging.set_verbosity(tf.logging.INFO)
    all_data, all_labels, all_dataset_map = get_relevant_data('train', config)
    spike_classifier = get_spike_classifier(config)

    if config['refine_recording'] is not None:
        # Copy directory
        dir_name = config['model_name'].rstrip('/')
        shutil.copytree(dir_name, dir_name + '_backup')

    # Split the data into a train an a test set.
    train_data, train_labels, eval_data, eval_labels = split_data(
        all_data, all_labels, all_dataset_map, config)

    eval_data, eval_labels = pad_to_batch_size(eval_data,
        eval_labels,
        config['batch_size'],
        config['N_neurons'])

    if config['niter'] > 0:
        # Set up logging for predictions
        def logging_hook():
            tensors_to_log = {"relu_coarse": "relu_coarse",
                              "zero_fraction_tracker": "zero_fraction_tracker"}
            return tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=200)

        # Create the Estimator
        validation_monitor = learn.monitors.ValidationMonitor(
            input_fn=learn.io.numpy_io.numpy_input_fn(
                x=eval_data,
                y=eval_labels,
                batch_size=config['batch_size'],
                num_epochs=1,
                shuffle=False
            ),
            every_n_steps=config['eval_every'],
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=config['early_stopping_rounds'])

        print("Fitting with batch size of %d" % config['batch_size'])
        spike_classifier.fit(
            input_fn=learn.io.numpy_io.numpy_input_fn(
                x=train_data,
                y=train_labels,
                batch_size=config['batch_size'],
                num_epochs=None,
                shuffle=False
            ),
            steps=config['niter'],
            monitors=[logging_hook(), validation_monitor])

    if config['refine_recording'] is not None:
        # Copy directory
        dir_name = config['model_name'].rstrip('/')
        shutil.move(dir_name,
                    '%s_recording%02d' % (dir_name, config['refine_recording']))
        shutil.move(dir_name + '_backup',
                    dir_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--global_features', type=str)
    parser.add_argument('--niter', type=int)
    parser.add_argument('--refine_recording', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--N_global_features', type=int)
    parser.add_argument('--validation_cycle', type=int)
    args = parser.parse_args()

    for key, val in vars(args).items():
        # Overwrite config with new vals
        if val is not None:
            config[key] = val

    tf.app.run()
