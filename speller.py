import time
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from attender import Attender
from decoder import char_to_num


class Speller(nn.Module):
    def __init__(self, CS, VOC, HFS, EMB, tfr):
        """
          In this network, we iterate over the time dimension of the input
          and thus each component only takes a single time step of data.
          Parameters:
            CS:     number of features in the context
            VOC:    number of output classes
            HFS:    number of hidden units in the speller
            EMB:    embedding size
            tfr:    teacher forcing rate
        """
        super(Speller, self).__init__()
        self.CS = CS
        self.VOC = VOC
        self.HFS = HFS
        self.EMB = EMB
        self.tfr = tfr
        
        # embedding layer
        self.embedding = nn.Embedding(VOC, EMB)
        # in: (BS), values in [0, VOC-1]
        # out: (BS, EMB)

        # First LSTM cell
        self.lstm_cell_a = nn.LSTMCell(EMB + CS, HFS, bias=True)
        # in: (BS, EMB+CS)
        #   hidden and cell: (BS, HFS)
        # out: 
        #   hidden and cell: (BS, HFS)

        # Second LSTM cell
        self.lstm_cell_b = nn.LSTMCell(HFS, HFS, bias=True)
        # in: (BS, HFS)
        #   hidden and cell: (BS, HFS)
        # out: 
        #   hidden and cell: (BS, HFS)
        
        # attention network
        self.attender = Attender(CS)
        # in: 
        #   key: (BS, RAL, CS)
        #   val: (BS, RAL, CS)
        #   query: (BS, CS)
        #   mask: (BS, RAL)
        # out: 
        #   context: (BS, CS)
        #   attention: (BS, RAL)

        # query projection
        self.query_proj = nn.Linear(HFS, CS)
        # in: (BS, HFS)
        # out: (BS, CS)

        # linear layer with softmax to get character distributions
        self.char_distr = nn.Linear(HFS + CS, VOC)
        # in: (BS, HFS+CS)
        # out: (BS, CS)
        self.softmax = nn.LogSoftmax(dim=-1)
        # in: *
        # out: *

    def forward(self, key, val, y, mask, tfr=0.9, pred_mode=False, log_att=False):
        """
          Forward pass of the speller.
          Parameters:
            key:        (BS, RAL, CS)
            val:        (BS, RAL, CS)
            y:          (BS, LAL)
            mask:       (BS, RAL)
            tfr:        teacher forcing rate, float scalar
            pred_mode:  whether or not in prediction mode
            log_att:    whether or not to log the attention
          Returns:
            pred:   prediction probabilities
                    (BS, LAL, VOC)
        """

        # make sure that y is only not given if we predict
        assert (y is not None or pred_mode)

        # save dimensions (250 is the max length)
        BS = key.size(0)
        LAL = 250 if y is None else y.size(1)

        # store our predictions
        pred = []

        # store the attention
        attentions = []

        # initial values for LSTM cell states and context
        state_a = None
        state_b = None
        context = None
        y_next = None

        # we now iterate through time in order to make our predictions
        for t in range(0, LAL):

            # decide if we do teacher forcing
            do_tf = random.random() > self.tfr or y is None

            # never do teacher forcing in prediction mode
            do_tf = False if pred_mode else do_tf

            # getting current time target
            if do_tf and t > 0:
                # initialize time target to last generated value
                y_next = F.softmax(y_next, dim=1)
                # y_next: (BS, VOC)

                # sample current y from last prediction
                y_t = Categorical(y_next).sample()
                # y_t: (BS) - values in [0, VOC-1]

            elif do_tf and t == 0:
                # feed it the start of sentence symbol <sos>
                sos_symbol = char_to_num['<sos>']
                y_t = np.array([sos_symbol] * BS)
                y_t = torch.Tensor(y_t)
                # y_t: (BS) - values in [0, VOC-1]

            else:
                # initialize time target to target value 
                y_t = y[:, t]
                # y_t: (BS) - values in [0, VOC-1]
            
            # ensure right type for y_t
            y_t = y_t.type(torch.LongTensor)
            if torch.cuda.is_available():
                y_t = y_t.cuda()

            embedding = self.embedding(y_t)
            # embedding: (BS, EMB)

            # create context if it doesn't exist
            if context is None:
                context = torch.zeros((BS, self.CS))
                context = context.cuda() if torch.cuda.is_available() else context
            # context: (BS, CS)

            # (BS, EMB) ++ (BS, CS) 
            lstm_in = torch.cat([embedding, context], dim=1)
            # lstm_in: (BS, EMB+CS)

            # run through the first cell
            hidden_a, cell_a = self.lstm_cell_a(lstm_in, state_a)
            # hidden_a: (BS, HFS)
            # cell_a: (BS, HFS)

            # pass previous hidden state to this as input
            hidden_b, cell_b = self.lstm_cell_b(hidden_a, state_b)
            # hidden_b: (BS, HFS)
            # cell_b: (BS, HFS)

            # hidden_b is the output of the RNN at this time
            # compute query from this
            query = self.query_proj(hidden_b)
            # query: (BS, CS)

            # compute context and attention using the attender
            context, attention = self.attender(key, val, query, mask)
            # context: (BS, CS)

            # save attention for first item only
            if log_att:
                att = attention.clone()
                attentions.append(att[0].cpu())

            # (BS, HFS) ++ (BS, CS)
            lstm_output = torch.cat([hidden_b, context], dim=1)
            # lstm_output: (BS, HFS+CS)

            # predictions for the next y
            y_next = self.char_distr(lstm_output)
            # y_next: (BS, VOC)

            # save the cell states for next iteration
            state_a = (hidden_a, cell_a)
            state_b = (hidden_b, cell_b)
            # state_a: (BS, HFS), (BS, HFS)
            # state_b: (BS, HFS), (BS, HFS)

            # append the prediction to our list
            pred.append(y_next)
        
        # make pred a tensor
        pred = torch.stack(pred, dim=1)
        # pred: (BS, LAL, VOC)

        # log attention
        if log_att:
            attentions = torch.stack(attentions)
            attentions = attentions.detach().numpy().transpose()
            plt.imshow(attentions)
            plt.savefig('attention/attention_{}'.format(int(time.time())))
            plt.close()

        # delete the context
        context = context.cpu()
        del context

        return pred
