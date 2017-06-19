# Build with:
#   sudo docker build -t patrickmineault:spikefinder .
# Run with
#   sudo docker run -it patrickmineault:spikefinder
#
# Note: I can't figure out how to make this run on GPU.
# nvidia-docker is of no help
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker
#
# See here for what I'm running into:
# https://github.com/tensorflow/tensorflow/issues/808
#
# Learning is quite slow on CPU.
# Therefore, this file only does inference using a pre-learned model.
FROM gcr.io/tensorflow/tensorflow

COPY . /usr/src/app

WORKDIR /usr/src/app

# This learns the long range features
# First clone the dependencies.
# git clone https://github.com/patrickmineault/write-rnn-tensorflow.git write_rnn_tensorflow
# CMD python train_long_range.py

# This learns the main model
# CMD ./fit_and_refine.sh

# And this does inference.
CMD ./predict_all.sh
