import torch.nn as nn


class pBLSTMLayer(nn.Module):
    def __init__(self, AUF, HFL):
        """
          AUF:  number of features for each time step
          HFL:  number of hidden units in the LSTM 
        """
        super(pBLSTMLayer, self).__init__()
        # bidirectional LSTM with one layer
        self.blstm = nn.LSTM(AUF*2, HFL, 1, bidirectional=True, batch_first=True)
        # in: (BS, X, 2*AUF)
        # out: (BS, X/2, 2*HFL)

    def forward(self, x, true_lens):
        """
          Forward pass through the pBLSTM layer. Pooling done before forward pass.
          Params:
            x:          (BS, AUL, AUF)
            true_lens:  true length of each tensor in batch
                        [BS]
          Returns: 
            x:          (BS, AUL/2, 2*HFL)
            true_lens:  updated lengths of each tensor in batch
                        [BS]
        """
        BS, AUL, AUF = x.size()
        # x: (BS, AUL, AUF)
        
        if AUL % 2 == 1:
            print('Warning: length of x not divisible by 2 (Listener)')
        
        # resize the input vector according to the pooling
        x = x.contiguous().view(BS, AUL // 2, 2*AUF)
        # x: (BS, AUL/2, 2*AUF)
        
        # update the true lengths
        true_lens = [e // 2 for e in true_lens]
        
        # bidirectional lstm
        x, _ = self.blstm(x)
        # x: (BS, AUL/2, 2*HFL)
        
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
            layer = pBLSTMLayer(2 * HFL, HFL)
            self.pyramidalBLSTM.append(layer)
            # in: (BS, X, 2*HFL)
            # out: (BS, X/2, 2*HFL)
        
        # two MLPs to get the attention key and value
        self.key_mlp = nn.Linear(2 * HFL, CS)
        self.val_mlp = nn.Linear(2 * HFL, CS)
        # in: (BS, 2*HFL)
        # out: (BS, CS)
    
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
            x:          (BS, AUL, AUF)
            true_lens:  list keeping track of the true length
                        of each input in the tensor x
                        [BS]
          Intermediate values:
            x:          Listener features
                        (BS, RAL, 2*HFL)
          Returns:
            key:        (BS, RAL, CS)
            val:        (BS, RAL, CS)
            true_lens:  keeping track of the new true
                        length of each tensor in the batch.
                        [BS]
        """
        # x: (BS, AUL, AUF)
        x, _ = self.base_blstm(x)     
        # x: (BS, AUL, 2*HFL)
        
        # each iteration halves the length of the tensor
        for layer in self.pyramidalBLSTM:
            x, true_lens = layer(x, true_lens)
        # x: (BS, RAL, 2*HFL)
        
        # get the key and value through MLPs
        key = self.key_mlp(x)
        val = self.val_mlp(x)
        # key: (BS, RAL, CS)
        # val: (BS, RAL, CS)

        return key, val, true_lens
