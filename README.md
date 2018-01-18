# Spikefinder submission

Reproduces my final submission to Spikefinder.

Steps to retrain model from scratch:

  * Clone this repo
  * `cd` into `spikefinder_submission` and clone mixture density network model
  via `git clone https://github.com/patrickmineault/write-rnn-tensorflow.git write_rnn_tensorflow`
  * Train mixture density network latent features via `python train_and_eval_mdn.py --operation=train`
  * Eval mixture density network latent features via `python train_and_eval_mdn.py --operation=eval`
  * Fit core model and refine for each of the first 5 recording sets by running `./train_and_refine.sh`
  * Eval for the unrefined and the refined model via `./eval_all.sh`

Steps to predict using pretrained model via Docker.

  * Clone this repo.
  * `sudo docker build -t patrickmineault:spikefinder` .
  * `sudo docker run -it patrickmineault:spikefinder`
