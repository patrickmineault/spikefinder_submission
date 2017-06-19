# Spikefinder submission

Reproduces my final submission to Spikefinder.

Steps to retrain model from scratch:

  * Clone this repo
  * `cd` into `spikefinder_submission` and clone mixture density network model
  via `git clone https://github.com/patrickmineault/write-rnn-tensorflow.git write_rnn_tensorflow`
  * Train mixture density network long-range features via `python train_long_range.py`
  * Fit core model and refine for each of the first 5 recording sets by running `./fit_and_refine.sh`
  * Predict for the unrefined and the refined model via `./predict_all.sh`

Steps to predict using pretrained model via Docker.

  * Clone this repo.
  * `sudo docker build -t patrickmineault:spikefinder` .
  * `sudo docker run -it patrickmineault:spikefinder`
