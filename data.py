import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class WSJDataset(Dataset):
    def __init__(self, data_path, data_prefix, data_format='l', sorting=True):
        """
          Parameters:
            data_path:      path to the folder containing all the data
            data_prefix:    prefix for the data we are using, possible: train / dev
            data_format:    which format to load the data in, possible: 
                            'w' for words, 'l' for characters, 'raw' for raw
          Variables:
            self.X:         list of numpy arrays, each of size (L x 40)
            self.Y:         list of numy arrays of words, each of size (L)
        """
        loader = lambda s: np.load(s, encoding='bytes')
        x_file = os.path.join(data_path, data_prefix + '.npy')
        y_file = os.path.join(data_path, data_prefix + '_transcripts_' + data_format + '.npy')
        self.X = loader(x_file)
        self.y = loader(y_file)
        # sorting the data
        if sorting:
            zipped = zip(*sorted(zip(self.X, self.y), key=lambda x: x[0].shape[0]))
            self.X, self.Y = (list(l) for l in zipped)
        # turning the numpy arrays into tensors
        for i in range(len(self.X)):
            self.X[i] = torch.Tensor(self.X[i])
        for i in range(len(self.y)):
            self.y[i] = torch.Tensor(self.y[i])
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.X)

def collate_lines(l):
    """
      Called by the data loader when collating lines. 
      Pad the sequences with zeros to turn them into a 3D tensor.
    """
    x, y = zip(*l)
    x, y = list(x), list(y)
    # padding
    x = rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = rnn.pad_sequence(y, batch_first=True, padding_value=0)
    # make sure the sequence lengths for the input are all 
    # multiples of 8 because of the pBLSTMs.
    if x.size(1) % 8 != 0:
        pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_len))
    return x, y

def train_loader():
    train_dataset = WSJDataset('data', 'train', sorting=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_lines)
    return train_loader

def val_loader():
    val_dataset = WSJDataset('data', 'dev', 'l', sorting=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_lines)
    return val_loader