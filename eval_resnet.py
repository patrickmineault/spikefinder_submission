"""Predict spikes from calcium models using pretrained models.
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

from train_resnet import get_relevant_data, get_spike_classifier
from util.nnio import eval_and_save, load_all, split_data, pad_to_batch_size
from util.nncomponents import dense_batch_relu, summarize_layer
import resnet

config = resnet.get_config()

def main(unused_argv):
    train_data, train_labels, train_dataset_map = get_relevant_data('train', config)
    train_data, train_labels = pad_to_batch_size(train_data,
        train_labels,
        config['batch_size'],
        config['N_neurons'])
    spike_classifier = get_spike_classifier(config)

    output_dir = 'preds/' + '_'.join(config['model_name'].split('/'))
    if config['refine_recording'] is not None:
        output_dir += '_refined'
    eval_and_save(spike_classifier,
                  train_data,
                  train_dataset_map,
                  output_dir,
                  '%d.train.spikes.csv',
                  config['batch_size'])

    test_data, test_labels, test_dataset_map = get_relevant_data('test', config)
    test_data, test_labels = pad_to_batch_size(test_data,
                                               test_labels,
                                               config['batch_size'],
                                               config['N_neurons'])
    eval_and_save(spike_classifier,
                  test_data,
                  test_dataset_map,
                  output_dir,
                  '%d.test.spikes.csv',
                  config['batch_size'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--refine_recording', type=int)
    args = parser.parse_args()

    for key, val in vars(args).items():
        # Overwrite config with new vals
        if val is not None:
            config[key] = val

    tf.app.run()
