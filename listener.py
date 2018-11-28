import torch.nn as nn


class pBLSTMLayer(nn.Module):
    def __init__(self, AUF, HFL):
        """
          AUF:  number of features for each time step
          HFL:  number of hidden units in the LSTM 
                output of LSTM will have 2*HFL features because 
                it's a bidirectional LSTM
        """
        super(pBLSTMLayer, self).__init__()
        # bidirectional LSTM with one layer
        self.blstm = nn.LSTM(AUF*2, HFL, 1, bidirectional=True, batch_first=True)

    def forward(self, x, true_lens):
        """
          Forward pass through the pBLSTM layer. Pooling done before forward pass.
          Assumptions:
            We assume that the input tensor is of a length divisible by two.
          Params:
            x:          input tensor of size (BS, AUL, AUF)
            true_lens:  list of length BS with true length of each tensor in batch
          Resized input size:
            (BS, AUL/2, 2*AUF)
          Returns: 
            x:          output tensor of size (BS, AUL/2, 2*HFL)
            true_lens:  list of length BS with updated lengths of each tensor in batch
        """
        # extract the dimensions
        BS, AUL, AUF = x.size()
        # make sure input tensor length is divisible by two
        assert AUL % 2 == 0
        # resize the input vector according to the pooling
        x = x.contiguous().view(BS, AUL // 2, 2*AUF)
        # update the true lengths
        true_lens = [e // 2 for e in true_lens]
        # bidirectional lstm
        x, _ = self.blstm(x)
        return x, true_lens


class Listener(nn.Module):
    def __init__(self, AUF, HFL, CS, n_lay=3):
        """
          Parameters:
            AUF:    number of features for each time step
            HFL:    number of hidden units in the LSTMs
            CS:     number of features in context for key and value
            n_lay:  number of pyramidal BLSTMs in the listener
        """
        super(Listener, self).__init__()
        # base BLSTM before the pyramidal BLSTMs
        self.base_blstm = nn.LSTM(AUF, HFL, bidirectional=True, batch_first=True)
        # pyramidal bidirectional LSTMs
        self.pyramidalBLSTM = nn.ModuleList()
        for _ in range(n_lay):
            # since it's bidirectional, it takes 2H features and returns 2H features
            # i.e. doesn't change number of features
            layer = pBLSTMLayer(2 * HFL, HFL)
            self.pyramidalBLSTM.append(layer)
        # two MLPs to get the attention key and value
        self.key_mlp = nn.Linear(2 * HFL, CS)
        self.val_mlp = nn.Linear(2 * HFL, CS)
    
    def forward(self, x, true_lens):
        """
          Forward input through three BLSTMs with pooling to reduce dimensions
          eight-fold, while keeping track of the length of each sequence.
          This module takes in an utterance of sequence length AUL with AUF
          features and computes listener features of sequence length RAL with
          2*HFL features.
          Reduced sequence length: RAL = AUL / (2^n_lay).
          It outputs the listener features as a key and a value, each 
          of dimensions (BS, RAL, CS) (through an MLP).
          Params:
            x:          Tensor of size (BS, AUL, AUF)
            true_lens:  list of length BS, keeping track of the true length
                        of each input in the tensor x
          Intermediate values:
            x:          Listener features, tensor of size (BS, RAL, 2*HFL)
          Returns:
            key:        Tensor of size (BS, RAL, CS)
            val:        Tensor of size (BS, RAL, CS)
            true_lens:  list of length BS, keeping track of the new true
                        length of each tensor in the batch.
        """
        # x is of size BS x AUL x AUF, true_lens doesn't change
        x, _ = self.base_blstm(x)        
        # x is of size B x AUL x 2*HFL
        for layer in self.pyramidalBLSTM:
            # each iteration halves the length of the tensor
            x, true_lens = layer(x, true_lens)
        # x is now of size BS x RAL x 2*HFL
        # get the key and value through MLPs
        # in BS x RAL x 2*HFL -> out BS x RAL x CS
        key = self.key_mlp(x)
        val = self.val_mlp(x)
        return key, val, true_lens
