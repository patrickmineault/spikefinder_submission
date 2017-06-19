from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow import summary

def scaled_xavier(weight=1.0):
    return tf.contrib.layers.variance_scaling_initializer(factor=weight,
                                                          mode='FAN_AVG',
                                                          uniform=False)

def summarize_layer(name, layer):
    with tf.name_scope(name + '_summary'):
        m = tf.reduce_mean(layer)
        tf.summary.scalar('mean', m)
        tf.summary.scalar('min', tf.reduce_min(layer))
        tf.summary.scalar('max', tf.reduce_max(layer))
        tf.summary.scalar('std', tf.sqrt(tf.reduce_mean((layer - m) ** 2)))
        tf.summary.histogram('histogram', layer)

# From http://ruishu.io/2016/12/27/batchnorm/
def dense_batch_relu(x, phase, scope, nunits=16, use_relu=False,
                     weight_scale=1.0, constant_offset=0.0):
  with tf.variable_scope(scope):
    h1 = tf.layers.dense(x,
                       nunits,
                       activation=None,
                       name='dense',
                       kernel_initializer=scaled_xavier(weight_scale),
                       bias_initializer=tf.constant_initializer(constant_offset)
                       )
    summarize_layer(scope, h1)

    h2 = tf.layers.batch_normalization(h1,
                                      training=phase,
                                      name='bn')
    if use_relu:
        h2 = tf.nn.relu(h2, 'relu')
    return h2

# From http://ruishu.io/2016/12/27/batchnorm/
def conv1d_batch_norm(x,
                      nunits,
                      window_size,
                      phase,
                      scope,
                      use_relu=False,
                      weight_scale=1.0,
                      dropout=0):
  with tf.variable_scope(scope):
    h1 = tf.layers.conv1d(
        x,
        nunits,
        window_size,
        strides=1,
        use_bias=True,
        kernel_initializer=scaled_xavier(weight_scale),
        bias_initializer=scaled_xavier(weight_scale),
        padding='valid',
        name='h1',
        activation=None)
    summarize_layer(scope, h1)

    h2 = tf.layers.batch_normalization(h1,
                                      training=phase,
                                      name='bn')
    if dropout > 0:
        h2 = tf.layers.dropout(h2,
                                    rate=dropout,
                                    training=phase)

    if use_relu:
        h2 = tf.nn.relu(h2, 'relu')
    return h2

def residual_layer(x, phase, scope, weight_scale=1.0):
    return x + dense_batch_relu(x, phase, scope, x.get_shape()[1], True, weight_scale)
