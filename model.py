import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

"""
Dimensions:
- B:    batch size
- N:    number of features
- H:    number of hidden features in the listener
- K:    number of input faetures for the preprocessing MLP
            (enconder state and listener features)
        number of hidden features in the speller
        K = 2*H
- L:    number of time steps
- U:    number of time steps after pooling with the pbLSTM
        U = L / 2^n, where n is the number of bLSTMs in the listener
- P:    number of features for attention (after the preprocessing MLP)
            (encoder state and listener features)
- C:    number of output classes
- T:    number of characters in the label/result
"""

class pBLSTMLayer(nn.Module):
    def __init__(self, N, H):
        """
          N: number of features for each time step
          H: number of features in the LSTM (will be multiplied by two because it's bidirectional)
        """
        super(pBLSTMLayer, self).__init__()
        # bidirectional LSTM with one layer
        self.blstm = nn.LSTM(N*2, H, 1, bidirectional=True, batch_first=True)

    def forward(self, x):
        """
          Forward pass through the pBLSTM layer. Pooling done before forward pass.
          Input size:
            B x L x N
          Resized input size:
            B x L/2 x 2*N
          Output size:
            B x L/2 x 2*H
        """
        # extract the dimensions
        B, L, N = x.size()
        # resize the input vector according to the pooling
        x = x.contiguous().view(B, L // 2, 2*N)
        # bidirectional lstm
        return self.blstm(x)

class Listener(nn.Module):
    def __init__(self, N, H):
        """
          Parameters:
            N:  number of features for each time step
            H:  number of features in the LSTM (will be multiplied by two
                because it's bidirectional)
        """
        super(Listener, self).__init__()
        # CONSIDER: adding a base BLSTM before the pyramidal BLSTMs.
        # WARNING!  currently not in use
        self.base_blstm = nn.LSTM(N, H, bidirectional=True, batch_first=True)
        # three pyramidal bidirectional LSTM with one layer each
        self.pBLSTM1 = pBLSTMLayer(N, H)
        self.pBLSTM2 = pBLSTMLayer(2 * H, H)
        self.pBLSTM3 = pBLSTMLayer(2 * H, H)
        # two MLPs to get the attention key and value
    
    def forward(self, x):
        """
          Forward input through three BLSTMs with pooling to reduce 
          dimensionality eight-fold.
          This module takes in an utterance of sequence length L with N 
          features and outputs listener features of sequence length U 
          (with U < L), K features (with K = 2*H).
          Input size:
            B x L x N
          Output size:
            B x U x K
          Example:
            Let N = 1000 and H = 256 and U = 125 (3 layers because 2^3 = 8)
            Input:  64 x 1000 x  40 = 2.6M
            Output: 64 x  125 x 512 = 4.1M
        """
        x, _ = self.pBLSTM1(x)
        x, _ = self.pBLSTM2(x)
        x, _ = self.pBLSTM3(x)
        return x

class Attention(nn.Module):
    def __init__(self, K, P):
        """
          Parameters:
            K:  number of input features for the preprocessing MLPs
                    (encoder state and listener features)
                number of hidden features in the speller
            P:  number of features after running preprocessing 
                    (on encoder state and listener features)
        """
        super(Attention, self).__init__()
        self.K = K
        self.P = P
        self.phi = nn.Linear(K, P)
        self.psi = nn.Linear(K, P)
        self.softmax = nn.Softmax(dim=-1)
        self.activate = F.relu

    def forward(self, si, h):
        """
          Parameters:
            si: decoder state at time step i
                B x 1 x K
            h:  high level listener features by the pyramidal bLSTM
                B x U x K
          Output:
            a:  list of one tensor of size B x U
            c:  tensor of size B x K
        """
        # preprocess the decoder state to B x 1 x P
        si = self.phi(si)
        si = self.activate(si)

        # preprocess the listener features
        h_saved = h.clone()
        B, U, K = h.size()
        # reshape to B*U x K in order to run it through the MLP
        # for preprocessing (each time step individually)
        h = h.contiguous().view(-1, K)
        # preprocessing of listener features
        h = self.psi(h)
        # reshape back to B x U x P
        h = h.view(B, U, self.P)
        h = self.activate(h)

        # transpose h to B x P x U
        hT = h.transpose(1, 2)
        # B x 1 x P mult B x P x U -> B x 1 x U
        e = torch.bmm(si, h.transpose(1, 2))
        # squeeze to B x U
        e = e.squeeze(dim=1)

        # attention score by taking softmax
        # will be B x U
        a = self.softmax(e)
        # turn into B x U x 1
        a_ = a.unsqueeze(2)
        # turn into B x U x K
        a_ = a_.repeat(1, 1, K)

        # compute the context
        # h_saved and a_ are both B x U x K (element wise multiplication)
        c = h_saved * a_
        # sum along the U-axis to yield B x K tensor
        c = torch.sum(c, dim=1)

        # return attention and context
        return [a], c

class Speller(nn.Module):
    def __init__(self, C, K, P, max_label_len=77):
        """
          Parameters:
            C:  number of output classes
            K:  number of input features for the preprocessing MLP
                number of hidden features in the speller
            P:  number of features after the preprocessing MLP
        """
        super(Speller, self).__init__()
        self.C = C
        self.K = K
        
        # save the float type (cuda or not)
        self.float_type = torch.torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.max_label_len = max_label_len

        # LSTM network with one layer
        self.lstm = nn.LSTM(C+K, K, num_layers=1, batch_first=True)
        
        # attention network
        self.attention = Attention(K, P)

        # linear layer with softmax to get character distributions
        self.char_distr = nn.Linear(2*K, C)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, h, y, rate=0.9):
        """
          Parameters:
            h:  listener features (from the Listener)
                B x U x K
            y:  ground truth
                B x T
        """
        # determine if we force the teacher or not
        if y is None:
            rate = 0.0
        force_teacher = np.random.random_sample() < rate

        # extract batch size
        B = h.size(0)

        # create the output word numpy array: B x 1
        output_word = np.zeros((B, 1))
        # turn into B x 1 x 1 float tensor
        output_word = self.float_type(output_word)
        output_word = output_word.unsqueeze(2)
        # turn into long type
        output_word = output_word.type(torch.LongTensor)
        # create temporary zero tensor of size B x 1 x C
        x = torch.LongTensor(B, 1, self.C)
        x = x.zero_()
        # create a tensor of size B x 1 x C 
        # with ones for the first class and zeros for all other classes
        x = x.scatter_(-1, output_word, 1)
        x = x.type(self.float_type)
        # turn it into a float typed variable
        output_word = Variable(x)
        # float variable of size B x 1 x C
        output_word = output_word.type(self.float_type)

        # extract feature from listener features (size B x U x K)
        # take only first time instance -> B x 1 x K
        rnn_input_h = h[:,0:1,:]
        # input to rnn from output_word and rnn_input_h
        # tensor of size B x 1 x (K+C)
        rnn_input = torch.cat([output_word, rnn_input_h], dim=-1)

        pred_seq_raw = []
        hidden_state = None
        attention_record = []

        if (y is None) or (not force_teacher):
            # we use our own produced characters
            max_step = self.max_label_len
        else:
            # we take the ground truth characters
            max_step = y.size(1)
        
        for step in range(max_step):
            # run the rnn_input through the LSTM (given initial hidden_state)
            # rnn_output of size B x 1 x K
            # hidden_state of sizes: B x 1 x K, B x 1 x K 
            #   (hidden state and cell state)
            rnn_output, hidden_state = self.lstm(rnn_input, hidden_state)
            # attention score and context from rnn output and listener feature
            # attention_score is list of one tensor of size B x U
            # context is tensor of size B x K
            attention_score, context = self.attention(rnn_output, h)

            # concatenate the features from the output with the context
            # B x K and B x K -> B x 2*K
            concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
            
            # raw prediction (through MLP), tensor of size B x C
            # input of shape B x 2*K
            raw_pred = self.char_distr(concat_feature)
            # softmax to get probabilities
            raw_pred = self.softmax(raw_pred)
            
            # append to the sequence of raw predictions
            pred_seq_raw.append(raw_pred)

            # append attention score to sequence
            attention_record.append(attention_score)

            if force_teacher:
                # force the input for the next timestep, size B x 1 x T
                target = y[:, step]
                output_word = torch.zeros_like(raw_pred)
                for idx, i in enumerate(target):
                    output_word[idx, int(i)] = 1
                output_word = output_word.type(self.float_type)
                output_word = output_word.unsqueeze(1)
            else:
                # ## Case 0. raw output as input, size B x 1 x C
                # output_word = raw_pred.unsqueeze(1)

                ## Case 1. Pick character with max probability
                # output word with size B x C
                output_word = torch.zeros_like(raw_pred)
                # get the k best predictions (here: only one)
                best_raw_pred = raw_pred.topk(1)
                best_raw_pred_idxs = best_raw_pred[1]
                # set the class label to 1 for the argmaxed class
                for idx, i in enumerate(best_raw_pred_idxs):
                    output_word[idx,int(i)] = 1
                # output word with size B x 1 x C
                output_word = output_word.unsqueeze(1)

                # ## Case 2. Sample categotical label from raw prediction
                # # create a categorical distribution from the raw prediction
                # categorical_distr = Categorical(raw_pred)
                # # sample a word from the distribution
                # sampled_word = categorical_distr.sample()
                # # set the class of the sampled word to one, all others zero
                # output_word = torch.zeros_like(raw_pred)
                # for idx, i in enumerate(sampled_word):
                #     output_word[idx,int(i)] = 1
                # # turn output_word into B x 1 x C
                # output_word = output_word.unsqueeze(1)
            
            # resize context tensor to size B x 1 x K
            context = context.unsqueeze(1)
            # cat output_word and context to get rnn input, B x 1 x (K+C)
            rnn_input = torch.cat([output_word, context], dim=-1)

        return pred_seq_raw, attention_record

def createModel(N, H, C, K, P, max_label_len):
    listener = Listener(N, H)
    speller = Speller(C, K, P, max_label_len)
    return listener, speller