import data
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical

"""
TODO:
  implement custom packed sequences

Dimensions:
  BS:  BATCH_SIZE
    batch size.
  AUF: AUDIO_FEATURES
    number of audio features in the input.
  HFL: HIDDEN_FEATURES_LISTENER
    number of hidden units in the listener.
  AUL: AUDIO_LENGTH
    length of the input audio sequence batch.
  RAL: REDUCED_AUDIO_LENGTH
    length of the input audio sequence after the pBLSTM.
  CS:  CONTEXT_SIZE
    context size for the listener output and speller output.
  LAL: LABEL_LENGTH
    length of the text sequence that is our target.
  VOC: VOCAB_SIZE
    number of different words/characters in the vocabulary.
"""

class Symbols:
    SOS = 32
    EOS = 33


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


class Attention(nn.Module):
    def __init__(self, CS):
        """
          Parameters:
            CS: number of features in the context
        """
        super(Attention, self).__init__()
        self.CS = CS

    def forward(self, key, val, query, mask):
        """
          Parameters:
            key:    key output by listener, tensor of size (BS, RAL, CS)
            val:    value output by listener, tensor of size (BS, RAL, CS)
            query:  this is the decoder output, tensor of size (BS, CS)
            mask:   tensor for the mask (true lengths) of size (BS, RAL)
          Output:
            c:      context tensor of size (BS, CS)
            a:      TODO attention score, list of one tensor of size (BS, RAL)
        """

        # query, make it size (BS, CS, 1)
        query = query.unsqueeze(2)

        # energy is the product of key and query
        # (BS, RAL, CS) x (BS, CS, 1)
        # energy is of size (BS, RAL, 1)
        energy = torch.bmm(key, query)
        # energy is of size (BS, RAL)
        energy = energy.squeeze(2)

        # attention is of size (BS, RAL)
        # TODO does this work without masking it?
        attention = F.softmax(energy, dim=1)

        # multiply the mask with the attention, element-wise
        attention = attention * mask
        # normalize the attention
        attention = attention / attention.sum(dim=1, keepdim=True)

        # (BS, 1, RAL) x (BS, RAL, CS) = (BS, 1, CS)
        attention = attention.unsqueeze(1)
        context = torch.bmm(attention, val)
        # context of size (BS, CS)
        context = context.squeeze(1)

        # attention of size (BS, RAL)
        attention = attention.squeeze(1)

        return context
        # TODO
        # return context, attention


class Speller(nn.Module):
    def __init__(self, CS, VOC, HFS, EMB):
        """
          Parameters:
            CS:     number of features in the context
            VOC:    number of output classes
            HFS:    number of hidden units in the speller
            EMB:    embedding size
        """
        super(Speller, self).__init__()
        self.CS = CS
        self.VOC = VOC
        self.HFS = HFS
        self.EMB = EMB
        
        # save the float type (cuda or not)
        self.float_type = torch.torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # embedding layer
        self.embedding = nn.Embedding(VOC, EMB)

        # First LSTM cell
        # in (BS, EMB+CS), hidden (BS, HFS) and cell (BS, HFS)
        # out hidden (BS, HFS) and cell (BS, HFS)
        self.lstm_cell_a = nn.LSTMCell(EMB + CS, HFS, bias=True)

        # Second LSTM cell
        # in (BS, HFS), hidden (BS, HFS) and cell (BS, HFS)
        # out hidden (BS, HFS) and cell (BS, HFS)
        self.lstm_cell_b = nn.LSTMCell(HFS, HFS, bias=True)
        
        # attention network
        self.attention = Attention(CS)

        # query projection
        self.query_proj = nn.Linear(HFS, CS)

        # linear layer with softmax to get character distributions
        self.char_distr = nn.Linear(HFS + CS, VOC)
        # TODO does this work without masking it?
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, key, val, y, mask, tfr=0.9):
        """
          Parameters:
            key:    size (BS, RAL, CS)
            val:    size (BS, RAL, CS)
            y:      size (BS, LAL)
            mask:   size (BS, RAL)
            tfr:    teacher forcing rate, float
          Returns:
            pred:   predictions, size (BS, LAL, VOC)
        """

        # extract dimensions
        BS, LAL = y.size()

        # list of LAL tensors of size (BS, VOC)
        pred = []

        # initial values for LSTM cell states and context
        state_a = None
        state_b = None
        context = None

        # we now iterate through time in order to make our 
        # predictions, using the LSTMCells
        for t in range(0, LAL):
            # TODO teacher forcing

            # initialize time target to first target value 
            # tensor of size (BS), values in [0, VOC-1]
            y_t = y[:, t].type(LongTensor)
            if torch.cuda.is_available():
                y_t = y_t.cuda()

            # get embedding, (BS, EMB)
            embedding = self.embedding(y_t)

            # create context bc it doesn't exist, move to GPU if possible
            # size (BS, CS)
            if context is None:
                context = torch.zeros((BS, self.CS))
                context = context.cuda() if torch.cuda.is_available() else context

            # (BS, EMB) cat (BS, CS) = (BS, EMB+CS)
            lstm_in = torch.cat([embedding, context], dim=1)

            # run through the first cell
            # hidden and cell state are (BS, HFS)
            hidden_a, cell_a = self.lstm_cell_a(lstm_in, state_a)

            # pass previous hidden state to this as input
            # hidden and cell state are (BS, HFS)
            hidden_b, cell_b = self.lstm_cell_b(hidden_a, state_b)

            # compute query from the hidden state of the second 
            # LSTM cell - the output of the RNN at this time step
            # size (BS, CS)
            query = self.query_proj(hidden_b)

            # compute the context using the attention network
            # size (BS, CS)
            context = self.attention(key, val, query, mask)

            # concatenate to get the output
            # (BS, HFS) cat (BS, CS)
            # size (BS, HFS+CS)
            lstm_output = torch.cat([hidden_b, context], dim=1)

            # predict next y, size (BS, VOC)
            y_next = self.char_distr(lstm_output)

            # save the cell states
            state_a = (hidden_a, cell_a)
            state_b = (hidden_b, cell_b)

            # append the prediction to our list
            pred.append(y_next)
        
        # free up GPU?
        context = context.cpu()
        del context

        # make pred a tensor of size (BS, LAL, VOC)
        pred = torch.stack(pred, dim=1)

        return pred


class ListenAttendSpell(nn.Module):
    def __init__(self, AUF, HFL, CS, VOC, HFS, EMB, n_listener=3, tfr=0.9):
        """
          Create a new instance of the Listen, Attend, Spell model.
          Params:
            AUF:    number of features in audio input
            HFL:    number of hidden states in the listener
            CS:     number of features in context
            VOC:    number of different elements in vocabulary
            HFS:    number of hidden states in the speller
            EMB:    number of features for the embedding
          Optional params:
            n_listener:     number of layers of pyramidal bLSTM in listener
            tfr:            teacher forcing rate
        """
        super(ListenAttendSpell, self).__init__()
        self.speller = Speller(CS, VOC, HFS, EMB)
        self.listener = Listener(AUF, HFL, CS, n_lay=n_listener)
    
    def forward(self, x, y):
        """
          Forward the given batch through the LAS model.
          Params:
            x:  list of BS tensors of size (AUL, AUF)
            y:  list of target labels
          Returns:
            xp: predictions of size (BS, LAL, VOC)
        """
        # compute the true length of all items in the list
        true_lens = [e.size(0) for e in x]

        # pad x and make it a tensor
        # x of size (BS, AUL, AUF)
        x = pad_sequence(x, batch_first=True)

        # make sure the sequence lengths for the input are all 
        # multiples of 8 because of the pBLSTMs.
        if x.size(1) % 8 != 0:
            pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_len))

        # pass x through the listener
        # key, val of size (BS, RAL, CS)
        # true_lens a list of BS integers
        key, val, true_lens = self.listener(x, true_lens)

        # pad y and make it a tensor
        # y of size (BS, LAL)
        y = pad_sequence(y, batch_first=True, padding_value=Symbols.EOS)

        # make the mask
        # mask of size (BS, RAL)
        mask_size = (x.size(0), key.size(1))
        mask = torch.zeros(mask_size)
        mask = mask.type(torch.int)
        mask = mask.cuda() if torch.cuda.is_available() else mask
        for batch_idx, max_time in enumerate(true_lens):
            mask[batch_idx, :max_time] = 1
        
        # use the speller to make the predictions
        pred = self.speller(key, val, y, mask)

        # free up GPU?
        mask = mask.cpu()
        del mask

        # return predictions of size (BS, LAL, VOC)
        return pred
    
    def train_batch(self, x, y):
        """
          Train this batch and return the loss
          Params:
            x:  list of BS tensors of size (AUL, AUF)
            y:  list of target labels
          Returns:
            bl: batch loss
        """
        # compute the true length of all items in the list
        true_lens = [e.size(0) for e in x]

        # pad x and make it a tensor
        # x of size (BS, AUL, AUF)
        x = pad_sequence(x, batch_first=True)

        # make sure the sequence lengths for the input are all 
        # multiples of 8 because of the pBLSTMs.
        if x.size(1) % 8 != 0:
            pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # pad y and make it a tensor
        # y of size (BS, LAL)
        y = pad_sequence(y, batch_first=True, padding_value=Symbols.EOS)

        # turn the data into a variable types
        x = Variable(x)
        y = Variable(y, requires_grad=False)
        x = x.type(torch.FloatTensor)
        # move to GPU if available
        if self.use_gpu:
            x = x.cuda()
            y = y.cuda()
        
        # prepare for forward pass
        self.optimizer.zero_grad()

        # pass x through the listener
        # key, val of size (BS, RAL, CS)
        # true_lens a list of BS integers
        key, val, true_lens = self.listener(x, true_lens)

        # make the mask
        # mask of size (BS, RAL)
        mask_size = (x.size(0), key.size(1))
        mask = torch.zeros(mask_size)
        mask = mask.type(torch.int)
        mask = mask.cuda() if torch.cuda.is_available() else mask
        for batch_idx, max_time in enumerate(true_lens):
            mask[batch_idx, :max_time] = 1
        
        # use the speller to make the predictions
        # pred of size (BS, LAL, VOC)
        pred = self.speller(key, val, y, mask)

        # get the true prediction
        y = y.data.contiguous()
        y = y.type(torch.cuda.LongTensor) if self.use_gpu else y.type(torch.LongTensor)

        # setup the criterion
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda() if self.use_gpu else criterion

        # compute the loss
        loss = criterion(pred, y)

        # backward pass
        loss.backward()
        self.optimizer.step()
        batch_loss = loss.cpu().item()

        # free up GPU?
        mask = mask.cpu()
        del mask

        return batch_loss
    
    def train(self, config_path):
        """
          Train the LAS model based on the hyperparameters found 
          in the configuration file.
        """
        # load configurations
        self.conf = yaml.load(open(config_path, 'r'))
        self.name = self.conf['meta_params']['model_name']

        # data loaders
        self.val_loader = data.val_loader()
        self.train_loader = data.train_loader()

        # learning parameters
        params = [{'params': self.listener.parameters()}, {'params': self.speller.parameters()}]
        l_rate = self.conf['training_params']['learning_rate']
        self.n_epochs = self.conf['training_params']['n_epochs']

        # optimizer for learning
        self.optimizer = torch.optim.Adam(params, l_rate)

        # check GPU availability
        self.use_gpu = torch.cuda.is_available()

        # move networks to GPU if possible
        if self.use_gpu:
            self.speller = self.speller.cuda()
            self.listener = self.listener.cuda()
        
        #
        # run the epochs
        #
        for epoch_i in range(self.n_epochs):

            n_batches = len(self.train_loader)
            epoch_start = time.time()
            batch_start = None

            # Training
            for idx, (batch_data, batch_label) in enumerate(self.train_loader):

                # start stopwatch for the next iteration
                batch_start = time.time()

                # forward and backward pass for this batch
                batch_loss = self.train_batch(batch_data, batch_label)

                # end stop watch for iteration
                batch_end = time.time()
                
                print('\rEpoch {:02}\tBatch {:03}/{:03}\tLoss {:7.3f}\tDur {:5.3f}'.format(epoch_i+1, idx+1, n_batches, batch_loss, batch_end - batch_start), end='', flush=True)
            
            print('\rEpoch {:02} completed in {:5.3f}s'.format(epoch_i, time.time() - epoch_start))
            
            self.save(epoch_i)
    
    def save(self, add):
        """
          Save the current model to the path specified in the config file.
          Params:
            add:    number to add to the end of the model path
        """
        # create folder if it doesn't exist yet
        path = self.conf['meta_params']['model_folder']
        if not os.path.exists(path):
            os.makedirs(path)
        # get the paths to which we save the model
        speller_path = os.path.join(path, '{}_{}.speller'.format(self.name, add))
        listener_path = os.path.join(path, '{}_{}.listener'.format(self.name, add))
        # save the state dictionaries
        torch.save(self.speller.state_dict(), speller_path)
        torch.save(self.listener.state_dict(), listener_path)
    
    def load(self, add):
        """
          Load the model from the path specified in the config file.
          Params:
            add:    string to add to the end of the model path
        """
        path = self.conf['meta_params']['model_folder']
        assert os.path.exists(path), 'Path to model doesnt exist'
        # get the paths from where we load the model
        speller_path = os.path.join(path, '{}_{}.speller'.format(self.name, add))
        listener_path = os.path.join(path, '{}_{}.listener'.format(self.name, add))
        # make sure the files exist
        assert os.path.exists(speller_path), 'Speller path doesnt exist'
        assert os.path.exists(listener_path), 'Listener path doesnt exist'
        # load the state dictionaries into the models
        self.speller.load_state_dict(torch.load(speller_path))
        self.listener.load_state_dict(torch.load(listener_path))

