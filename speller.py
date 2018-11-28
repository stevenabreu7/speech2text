import torch
import torch.nn as nn
from attender import Attender


class Speller(nn.Module):
    def __init__(self, CS, VOC, HFS, EMB, tfr):
        """
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
        self.attender = Attender(CS)

        # query projection
        self.query_proj = nn.Linear(HFS, CS)

        # linear layer with softmax to get character distributions
        self.char_distr = nn.Linear(HFS + CS, VOC)
    
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
            y_t = y[:, t].type(torch.LongTensor)
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
            context = self.attender(key, val, query, mask)

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
