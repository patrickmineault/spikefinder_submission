# Train long-range features for the spikefinder submission.
# It takes in calcium traces and learns a mixture density network model for each
# for each calcium trace snippet. It then extracts features which are preserved
# for a given cell but which are discriminative across cells.
# This is an adaptation of the hardmaru's write-rnn-tensorflow/train.py

import argparse
import cPickle as pickle
import time
import os

import numpy as np
import scipy
import scipy.linalg
import scipy.signal
import tensorflow as tf

from util.nnio import load_csv
from util.load_as_handwriting import DataLoader
from write_rnn_tensorflow.model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                     help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=32,
                     help='minibatch size')
    parser.add_argument('--batch_size_deep', type=int, default=64,
                     help=('Batch size for the later deep neural net. Must be '
                     'a divisor of seq_length'))
    parser.add_argument('--features_to_keep', type=int, default=32,
                 help=('Number of features to keep after the SVD stage.'))
    parser.add_argument('--seq_length', type=int, default=1024,
                     help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=15,
                     help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                     help='save frequency')
    parser.add_argument('--model_dir', type=str, default='mdn_model',
                     help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0015,
                     help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
    parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=20,
                     help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale)
    assert data_loader.num_batches > 0
    assert data_loader.data[0].shape[1] == 3
    x, y = data_loader.next_batch()
    assert x[0].shape[1] == 3
    v_x, v_y = data_loader.validation_data()


    if args.model_dir != '' and not os.path.exists(args.model_dir):
      os.makedirs(args.model_dir)

    # Also add in directories to store the outputs of the models.
    if not os.path.exists('spikefinder.train.longrange'):
        os.makedirs('spikefinder.train.longrange')

    if not os.path.exists('spikefinder.test.longrange'):
        os.makedirs('spikefinder.test.longrange')

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    sess = tf.InteractiveSession()
    summary_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), sess.graph)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    for e in range(args.num_epochs):
        sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
        data_loader.reset_batch_pointer()
        v_x, v_y = data_loader.validation_data()
        valid_feed = {model.input_data: v_x, model.target_data: v_y, model.state_in: model.state_in.eval()}
        state = model.state_in.eval()

        state_means = []
        state_vars = []
        for b in range(data_loader.num_batches):
            i = e * data_loader.num_batches + b
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.target_data: y, model.state_in: state}
            train_loss_summary, train_loss, state, state_mean, state_var, _ = sess.run([model.train_loss_summary, model.cost, model.state_out, model.state_mean, model.state_var, model.train_op], feed)
            summary_writer.add_summary(train_loss_summary, i)

            valid_loss_summary, valid_loss, = sess.run([model.valid_loss_summary, model.cost], valid_feed)
            summary_writer.add_summary(valid_loss_summary, i)

            state_means.append(state_mean)
            state_vars.append(state_var)
            end = time.time()
            print(
                "{}/{} (epoch {}), train_loss = {:.3f}, valid_loss = {:.3f}, time/batch = {:.3f}"  \
                .format(
                    i,
                    args.num_epochs * data_loader.num_batches,
                    e,
                    train_loss, valid_loss, end - start))
            if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                print("model saved to {}".format(checkpoint_path))

        m = np.concatenate(state_means, axis=0)
        v = np.concatenate(state_vars, axis=0)
        state_importance = np.std(m / np.sqrt(np.mean(v, axis=0, keepdims=True)), axis=0)
        most_important = (-state_importance).argsort()
        targets = sorted(most_important[:64])
        print(targets)

    # Now that the model is fit, reduce the learned latent features to 32
    # and output.
    def next_batch_sequential(self, data):
        # returns a randomised, seq_length sized portion of the training data
        k = 0
        for i, cell in enumerate(data):
            n_batch = int((len(cell) - 1)/((self.seq_length))) # number of equiv batches this datapoint is worth
            for j in range(n_batch):
                    idx = j * self.seq_length
                    x = np.copy(cell[idx:idx+self.seq_length])
                    y = np.copy(cell[idx+1:idx+self.seq_length+1])
                    yield x, y, i, False
                    k = k + 1
        left_over = np.ceil(float(k) / float(self.batch_size)) * self.batch_size - k
        assert left_over >= 0
        for m in range(int(left_over)):
            yield x, y, i, True

    def get_data(source, state):
        if source == 'train':
            data = data_loader.data
        else:
            data = data_loader.valid_data
        xs = []
        ys = []
        is_ = []
        is_repeats = []
        state_means = []
        state_vars = []

        n = 0
        for x, y, i, is_repeat in next_batch_sequential(data_loader, data):
            xs.append(x)
            ys.append(y)
            is_.append(i)
            is_repeats.append(is_repeat)

            n += 1
            if n % data_loader.batch_size == 0:
                X = xs[-data_loader.batch_size:]
                Y = ys[-data_loader.batch_size:]
                feed = {model.input_data: X,
                        model.target_data: Y,
                        model.state_in: state}
                state, state_mean, state_var = sess.run(
                    [model.state_out, model.state_mean, model.state_var], feed)
                state_means.append(state_mean)
                state_vars.append(state_var)
        state_means = np.concatenate(state_means, axis=0)
        state_vars  = np.concatenate(state_vars, axis=0)
        neuron_nums = np.array(is_)
        is_repeats = np.array(is_repeats)

        state_means = state_means[~is_repeats, :]
        state_vars = state_vars[~is_repeats, :]
        neuron_nums = neuron_nums[~is_repeats]
        return state_means, state_vars, neuron_nums

    state_means, state_vars, neuron_nums = get_data('train', state)

    zs = []
    for i in range(neuron_nums.max()+1):
        z = (state_means[neuron_nums == i, :].mean(axis=0, keepdims=True) /
             np.sqrt(state_vars[neuron_nums == i, :].mean(0, keepdims=True)))
        zs.append(z)
    Z = np.concatenate(zs, axis=0)
    Zm = Z.mean(axis=0, keepdims=True)
    Z = Z - Zm

    U, S, Vh = scipy.linalg.svd(Z)
    projection_matrix = np.copy(Vh.T)[:, :args.features_to_keep]

    # Rescale the features.
    projection_matrix = projection_matrix * (
        1.0 / (.001 + np.sqrt(S[:args.features_to_keep].reshape((1, -1)))))

    # And now write it out to disk
    for dataset in ('train', 'test'):
        state_means, state_vars, neuron_nums = get_data(dataset, state)
        X, _ = load_csv(dataset)
        i = 0
        for recording_num, Xs in enumerate(X):
            for k, col in enumerate(Xs.columns):
                print(col)
                # We have to jump through some hoops because we use a batch size
                # of 1024 here but 64 when training the deep neural net model.
                ndups = int(args.seq_length / args.batch_size_deep)
                m = state_means[neuron_nums == i, :]
                m = m / np.sqrt(state_vars[neuron_nums == i].mean(0, keepdims=True))
                m = m - Zm
                p = m.dot(projection_matrix)
                p = np.tile(p.reshape((-1, 1, args.features_to_keep)), [1, ndups, 1])
                p = p.reshape(p.shape[0] * p.shape[1], p.shape[2])

                # Pad.
                p = np.concatenate((p, np.tile(p[-1:, :], [args.batch_size_deep, 1])), axis=0)

                # Smooth
                win = np.ones((ndups - 1, 1))

                p = (scipy.signal.convolve2d(p, win, 'same') /
                     scipy.signal.convolve2d(np.ones_like(p), win, 'same'))

                i += 1
                fname = 'spikefinder.%s.longrange/%s.%d.calcium.neuron.%s.batch-size-%s.cpkl' % (dataset, dataset, recording_num + 1, col, args.batch_size_deep)
                with open(fname, 'w') as f:
                    pickle.dump(p, f)

if __name__ == '__main__':
    main()
