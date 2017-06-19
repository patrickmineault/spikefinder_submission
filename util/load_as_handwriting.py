"""Load spikefinder data and format it so it looks like handwriting data.

This allows the write-rnn-tensorflow model to use this data directly. Hacky but
easier to adjust the data than to adjust the model.
"""
import numpy as np
import random
from nnio import load_csv

class DataLoader(object):
    def __init__(self, batch_size, seq_len, data_scale):
        self.batch_size = batch_size
        self.seq_length = seq_len
        self.data_scale = data_scale
        self.reset_batch_pointer()
        self.preprocess()

    def preprocess(self):
        def preprocess_one(source):
            counter = 0
            all_X, _ = load_csv(source)

            datasets = []
            for Xsub in all_X:
                for i, col in enumerate(Xsub.columns):
                    X = Xsub[col]
                    bad_idx = np.isnan(X)
                    nbad = bad_idx.sum()
                    nnas = (~bad_idx).sum()
                    X = X.loc[~bad_idx]
                    X = np.tile(X.values.reshape((-1, 1)), [1, 3])
                    X = X / X.std().reshape((1, -1))
                    X = X + .05*np.random.randn(X.shape[0], X.shape[1])
                    # Declare end-of-stroke at random intervals.
                    X[:, -1] = 1.0*(np.random.rand((X.shape[0])) < .01)
                    counter += int(X.shape[0]/((self.seq_length+2)))
                    datasets.append(X)
            return datasets, int(counter/ self.batch_size)
        self.data, self.num_batches = preprocess_one('train')
        self.data_shuffled = list(self.data)
        random.shuffle(self.data_shuffled)
        self.valid_data, _ = preprocess_one('test')
        self.num_datasets = len(self.data)

    def reset_batch_pointer(self):
        self.pointer = 0

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
          self.pointer = 0

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.valid_data[i%len(self.valid_data)]
            idx = 0
            x_batch.append(np.copy(data[idx:idx+self.seq_length]))
            y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
        return x_batch, y_batch

    def next_batch(self):
        # returns a randomised, seq_length sized portion of the training data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
          data = self.data_shuffled[self.pointer]
          n_batch = int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth
          idx = random.randint(0, len(data)-self.seq_length-2)
          x_batch.append(np.copy(data[idx:idx+self.seq_length]))
          y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
          if random.random() < (1.0/float(n_batch)): # adjust sampling probability.
            # if this is a long datapoint, sample this data more with higher probability
            self.tick_batch_pointer()
        return x_batch, y_batch
