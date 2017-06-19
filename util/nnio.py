from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import collections
import numpy as np
import os
import os.path
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import learn

def pad_to_boundary(y, boundary, val=0, type='zero'):
    if y.size % boundary != 0:
        if type == 'zero':
            y = np.hstack((y, val * np.ones(boundary - (y.size % boundary))))
        elif type == 'mirror':
            y = np.hstack((y, y[-1:-(boundary - (y.size % boundary) + 1):-1]))
        else:
            raise NotImplementedError("Unhandled boundary condition")
    assert y.size % boundary == 0
    return y


def compute_global_features(X, config):
    # Pad X left and right
    shape = X.shape
    X = X.reshape((-1,))
    ls = config['global_lengthscale']

    ntoskip = 0
    while ls > X.size:
        ntoskip += X.size
        X = np.hstack((X[-1::-1], X, X[-1::-1]))

    npaddings = int(np.ceil(ls / config['window_size']))

    pad_l = X[(npaddings * config['window_size']):0:-1]
    pad_r = X[-1:-(npaddings * config['window_size'] + 1):-1]

    X = np.hstack((pad_l, X, pad_r))

    df = pd.DataFrame({'X': X})
    df['Xd'] = df.X.diff()

    # Rolling averages
    def rolling_vals(r): return [r.mean().as_matrix(),
                                 r.quantile(.1).as_matrix(),
                                 r.quantile(.2).as_matrix(),
                                 r.quantile(.5).as_matrix(),
                                 r.quantile(.8).as_matrix(),
                                 r.quantile(.9).as_matrix(),
                                 r.std().as_matrix(),
                                 r.skew().as_matrix(),
                                 r.kurt().as_matrix()]

    X_stats = rolling_vals(df['X'].rolling(window=ls, center=True))

    # Autocorrelation stats
    autocorr_stats = []
    for i in [0, 1, 6, 11, 16, 21, 26, 31]:
        xcorr = ((df['X'] - X_stats[0]) *
                 (df['X'] - X_stats[0]).shift(i)).rolling(window=ls, center=True)
        x = xcorr.mean().as_matrix()
        xcorr = ((df['X'] - X_stats[0]).diff() *
                 (df['X'] - X_stats[0]).diff().shift(i)).rolling(window=ls, center=True)
        dx = xcorr.mean().as_matrix()

        if i == 0:
            x0 = x
            dx0 = dx
        else:
            autocorr_stats.append(x / x0)
            autocorr_stats.append(dx / dx0)

    dX_stats = rolling_vals(df['Xd'].rolling(window=ls, center=True))

    all_stats = X_stats + dX_stats + autocorr_stats
    X = np.vstack(all_stats).T
    if ntoskip > 0:
        X = X[ntoskip:-ntoskip, :]
    X = X.reshape((-1, config['window_size'], X.shape[1]))
    X = X.mean(1).squeeze()
    X = X[npaddings:-npaddings, :]
    assert X.shape[0] == shape[0]
    return X


def load_all(mode='train', num_datasets=1, config=None):
    all_X, all_y = load_csv(mode)

    N_batches = 0
    neuron_num = 0
    dfs = []
    ys = []

    dataset_map = {}
    for n in range(min(len(all_X), num_datasets)):
        for i, col in enumerate(all_X[n].columns):
            tf.logging.info(
                "Computing features for neuron num %d, %d" % (n, i))
            X = all_X[n][col]
            bad_idx = np.isnan(X)
            nbad = bad_idx.sum()
            nnas = (~bad_idx).sum()
            X = X.loc[~bad_idx]

            X0 = X[0:(config['window_size'] + 1):config['window_size']].as_matrix()
            if nbad > 0:
                # Check that missingness is only at the end of the file.
                assert bad_idx[:-nbad].sum() == 0

            if config['feature_type'] == 'continuous':
                # Start by padding to size.
                w = pad_to_boundary(np.ones(X.size),
                                    config['window_size'])
                neuron_ids = pad_to_boundary(neuron_num * np.ones(X.size),
                                             config['window_size'],
                                             config['N_neurons'])
                w = w.reshape((-1, config['window_size']))
                neuron_ids = neuron_ids.reshape((-1, config['window_size']))
                assert w[-1, 0] != 0
                assert w[-2, -1] != 0
                assert w[0, 0] != 0

                X = pad_to_boundary(
                    X, config['window_size'], type=config['boundary_conditions'])
                assert X.size % config['window_size'] == 0

                # Pad a certain number of times.
                max_padding = max([config['window_padding_l'], config['window_padding_r']])
                total_padding = config['window_padding_l'] + config['window_padding_r']
                initial_x = X[0]
                npaddings = int(np.ceil(
                    max_padding / config['window_size']))
                assert npaddings > 0 or max_padding == 0

                if config['boundary_conditions'] == 'mirror':
                    pad_l = X[(npaddings * config['window_size']):0:-1]
                    pad_r = X[-1:-(npaddings * config['window_size'] + 1):-1]

                    assert pad_l.size == (npaddings * config['window_size'])
                    assert pad_r.size == (npaddings * config['window_size'])

                    X = np.hstack((pad_l, X, pad_r))
                elif config['boundary_conditions'] == 'zero':
                    pad = np.zeros(npaddings * config['window_size'])
                    X = np.hstack((pad, X, pad))
                else:
                    raise NotImplementedError("Unknown boundary condition")

                X = X.reshape((-1, config['window_size']))

                if config['global_features'] == 'prepost':
                    global_features = compute_global_features(X, config)
                    global_features = global_features.astype(np.float32)
                    global_features = global_features[npaddings:-npaddings]

                assert npaddings == 0 or X[0, -
                                           1] == 0 or X[npaddings - 1, -1] == X[npaddings, 1]
                assert X[npaddings, 0] == initial_x


                # Concatenate left to right
                X_ = []
                for i in range(1 + 2 * npaddings):
                    X_.append(X[slice(i, i + X.shape[0] - 2 * npaddings)])
                X = np.concatenate(X_, axis=1)

                if config['global_features'] == 'prepostunsupervised':
                    fname = ("spikefinder.%s.longrange/%s.%d.calcium.neuron.%s.batch-size-%d.cpkl" % (mode, mode, n + 1, col, (config['window_size']/2)))
                    with open(fname) as f:
                        global_features = pickle.load(f)
                    global_features = global_features[:X.shape[0]].astype(np.float32)
                elif config['global_features'] != 'prepost':
                    global_features = np.zeros((X.shape[0], 1))

                delta_l = (npaddings * config['window_size'] - config['window_padding_l'])
                delta_r = (npaddings * config['window_size'] - config['window_padding_r'])

                if delta_l > 0:
                    X = X[:, delta_l:]
                if delta_r > 0:
                    X = X[:, :-delta_r]
                assert X.shape[1] == total_padding + config['window_size']
                assert X[0, config['window_padding_l']] == initial_x
                assert (config['window_padding_l'] == 0 or
                        X[0, config['window_padding_l'] - 1] == 0 or
                        X[0, config['window_padding_l'] - 1] == X[0, config['window_padding_l'] + 1])

                assert (config['window_padding_r'] == 0 or
                        X[0, config['window_size'] + config['window_padding_l']] == 0 or
                        X[0, config['window_size'] + config['window_padding_l']] == X[1, config['window_padding_l']])


                assert w.shape[0] == X.shape[0]
                assert w.shape[1] <= X.shape[1]
                assert global_features.shape[0] == X.shape[0]

                next_batch = np.arange(0, X.shape[0]) + 1
                next_batch += N_batches
                next_batch[-1] = -1

                prev_batch = np.arange(0, X.shape[0]) - 1
                prev_batch += N_batches
                prev_batch[0] = -1
                next_batch, prev_batch = (next_batch.reshape((-1, 1)),
                                          prev_batch.reshape((-1, 1)))

                state_bw = np.zeros((X.shape[0], config['state_size']), np.float32)
                state_fw = np.zeros((X.shape[0], config['state_size']), np.float32)

            w = config['mean_obs_per'] * w / w.sum()

            if mode == 'train':
                y = all_y[n][col]
                y0 = y[0:(config['window_size'] + 1):config['window_size']].as_matrix()

                y = all_y[n][col].loc[~bad_idx].as_matrix()
                ynorm2 = (y ** 2).sum()
                w = w * y.size / ynorm2

                # y is normalized so that the RMSE metric is equivalent to a
                if config['feature_type'] == 'continuous':
                    y = pad_to_boundary(y, config['window_size'])
                    w = pad_to_boundary(w, config['window_size'])

                assert np.isnan(y).sum() == 0
                y2 = y[config['window_size'] * 2 - 1]
                y = y.reshape((-1, config['window_size']))
                assert y[1, -1] == y2
                ys.append(y)

                assert y.shape[0] == X.shape[0]
                assert neuron_ids.shape == y.shape
                assert w.shape == y.shape
                assert global_features.shape[0] == y.shape[0]
                assert X[0, config['window_padding_l']] == X0[0]
                assert X[1, config['window_padding_l']] == X0[1]
                assert y[0, 0] == y0[0]
                assert y[1, 0] == y0[1]
                assert (w >= 0).all()

            dfs.append({'X': X.astype(np.float32),
                        'neuron_ids': neuron_ids.astype(dtype=np.int32),
                        'w': w.astype(np.float32),
                        'global_features': global_features[:, :config['N_global_features']],
                        'state_bw': state_bw,
                        'state_fw': state_fw,
                        'prev_batch': prev_batch,
                        'next_batch': next_batch,
                        #'T': np.arange(X.size).reshape((X.shape[0], X.shape[1], 1)).astype(np.float32),
                        'dirty': np.ones_like(prev_batch, dtype=np.float32),
                        'batch_id': np.zeros_like(prev_batch, dtype=np.float32),
                        'alphas': np.ones_like((state_bw))
                        })

            dataset_map[neuron_num] = (n, i, nnas)
            neuron_num += 1
            N_batches += X.shape[0]

    if ys:
        Y = np.concatenate(ys, axis=0).astype(np.float32)
    else:
        Y = None

    Xdict = {}
    for name in dfs[0].keys():
        Xdict[name] = np.concatenate([x[name] for x in dfs], axis=0)

    next_batch = Xdict['next_batch']
    next_batch = next_batch[next_batch != -1]
    assert np.unique(next_batch).size == next_batch.size
    assert np.all(next_batch != 0)
    assert next_batch.max() == Xdict['X'].shape[0] - 1

    prev_batch = Xdict['prev_batch']
    prev_batch = prev_batch[prev_batch != -1]
    assert np.unique(prev_batch).size == prev_batch.size
    assert prev_batch.min() == 0
    assert prev_batch.max() == Xdict['X'].shape[0] - 2


    return Xdict, Y, dataset_map


def subset_to_neuron(data, neuron_num):
    neuron_data = data['neuron_ids'].min(axis=1) == neuron_num
    return {k: v[neuron_data, :] for k, v in data.items()}


def subset_minibatch(dat, offset, batch_size):
    rg = np.arange(offset, offset + batch_size)
    rg = np.fmin(rg, dat['X'].shape[0] - 1)
    return {x: y[rg, :] for x, y in dat.items()}

def pad_to_batch_size(data, labels=None, batch_size=1, N_neurons=1):
    npad = (int(np.ceil(data['X'].shape[0] / batch_size) * batch_size) -
            data['X'].shape[0])
    if npad == 0:
        return data, labels
    for key, val in data.items():
        if key == 'w':
            pad = np.zeros((npad, val.shape[1]), dtype=val.dtype)
        elif key == 'neuron_ids':
            pad = N_neurons*np.ones((npad, val.shape[1]), dtype=val.dtype)
        else:
            pad = np.tile(val[-1:], [npad] + [1 for x in range(val.ndim-1)])
        data[key] = np.concatenate((val, pad), axis=0)
    if labels is not None:
        labels = np.concatenate((labels, np.zeros((npad, labels.shape[1]), dtype=labels.dtype)))
    return data, labels

def eval_and_save(spike_classifier,
                  data,
                  dataset_map,
                  output_dir,
                  output_pattern,
                  batch_size):
    tf.logging.info("Evaluating on " + output_pattern)

    # Reorganize into pandas data frames.
    datasets = collections.defaultdict(lambda: [])
    max_num = sorted(np.unique(data['neuron_ids'].ravel()))[-2]

    for i in range(max_num + 1):
        tf.logging.info("Evaluating on neuron num %d" % i)

        neuron_data = subset_to_neuron(data, i)
        if neuron_data['X'].size == 0:
            continue
        neuron_data, _ = pad_to_batch_size(neuron_data, None, batch_size, data['neuron_ids'].max())
        sz0 = neuron_data['X'].shape[0]
        ys = []
        y = spike_classifier.predict(
            input_fn=learn.io.numpy_io.numpy_input_fn(
                x=neuron_data,
                batch_size=batch_size,
                num_epochs=1,
                shuffle=False
            ))

        y = np.concatenate([x['relu_output'] for x in y], axis=0)
        y = y[neuron_data['neuron_ids'].ravel() == i]
        collection_num, _, nnas = dataset_map[i]
        assert y.size == nnas
        datasets[collection_num].append(y)
    try:
        os.makedirs(output_dir)
    except OSError:
        print("Directory already created.")

    for num, d in datasets.items():
        max_size = max([x.size for x in d])

        # Pad columns with NA so they're all the same size
        # Also, normalize.
        d2 = {i: np.hstack((x.squeeze() / (x.std() + .001), np.nan *
                            np.ones(max_size - len(x)))) for i, x in enumerate(d)}

        # And save where appropriate.
        name = os.path.join(output_dir, output_pattern % (num + 1))
        df = pd.DataFrame(d2)
        df.to_csv(name, index=False, float_format="%.4f")

kfolds = 5
def split_data(all_data, all_labels, all_dataset_map, config):
    if not config['validation_bycell']:
        # Validate every 5th batch
        train_idx = np.floor(
            np.arange(all_labels.shape[0]) / config['validation_cycle']) % kfolds != 0
    elif config['validation_bycell'] == 'sub':
        neuron_idx = all_data['neuron_ids'][:, 0]
        train_idx = np.ones(neuron_idx.shape, dtype=np.bool_)

        for i in range(neuron_idx.max() + 1):
            ns = np.where(neuron_idx == i)[0]
            if len(ns) > 0:
                assert (ns[-1] - ns[0] + 1 == ns.size)
                j = i % kfolds
                lo_idx = int( (j / float(kfolds)) * ns.size )
                hi_idx = int( ((j + 1) / float(kfolds)) * ns.size )
                train_idx[ns[lo_idx:hi_idx]] = False
    else:
        # Validate every 5th cell
        train_idx = all_data['neuron_ids'][:, 0] % kfolds != 0

    # Make sure there's the expected train/validation split.
    assert abs(train_idx.sum() / float(train_idx.size ) - (kfolds - 1) / float(kfolds)) < .01

    train_labels = all_labels[train_idx, :]
    shuffler = np.random.permutation((np.arange(train_labels.shape[0])))
    train_labels = train_labels[shuffler, :]
    train_data = {k: v[train_idx, :][shuffler, :]
                  for k, v in all_data.items()}

    eval_labels = all_labels[~train_idx, :]
    shuffler = np.random.permutation((np.arange(eval_labels.shape[0])))
    eval_labels = eval_labels[shuffler, :]
    eval_data = {k: v[~train_idx, :] for k, v in all_data.items()}
    eval_data = {k: v[shuffler, :] for k, v in eval_data.items()}

    return train_data, train_labels, eval_data, eval_labels

def load_csv(dataset="train"):
    """load_csv reads data from either the train or test CSVs

    Args:
      dataset: either "train" or "test"

    Returns:
      tuple of X, y. Each X corresponds to one recording session. Each column
      of each X is one electrode.
    """
    X = []
    y = []
    if dataset == 'train':
        for d in range(1, 11):
            X.append(pd.read_csv("spikefinder.train/%d.train.calcium.csv" % d))
            y.append(pd.read_csv("spikefinder.train/%d.train.spikes.csv" % d))
    elif dataset=='test':
        for d in range(1, 6):
            X.append(pd.read_csv("spikefinder.test/%d.test.calcium.csv" % d))
            y.append(None)
    return X, y
