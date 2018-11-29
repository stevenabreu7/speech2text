import torch
import torch.nn as nn
import torch.nn.functional as F


class Attender(nn.Module):
    def __init__(self, CS):
        """
          Parameters:
            CS: number of features in the context
        """
        super(Attender, self).__init__()
        self.CS = CS

    def forward(self, key, val, query, mask):
        """
          The listener outputs the key and value. The query
          is the output of the custom LSTM inside the speller.
          The mask shows the true lengths of the input sequences.
          This method computes the attention from these values 
          and uses it to compute the context.
          The speller will continue working with the context,
          the attention is returned only for debugging reasons.
          Parameters:
            key:    key output by listener
                    (BS, RAL, CS)
            val:    value output by listener
                    (BS, RAL, CS)
            query:  this is the decoder output
                    (BS, CS)
            mask:   mask with true lengths
                    (BS, RAL)
          Output:
            cont:   context
                    (BS, CS)
            att:    attention
                    (BS, RAL)
        """

        query = query.unsqueeze(2)
        # query: (BS, CS, 1)

        # energy is the product of key and query
        # (BS, RAL, CS) x (BS, CS, 1)
        energy = torch.bmm(key, query)
        # energy: (BS, RAL, 1)
        energy = energy.squeeze(2)
        # energy: (BS, RAL)

        # take the softmax
        attention = F.softmax(energy, dim=1)
        # attention: (BS, RAL)

        # make sure mask is correct type
        mask = mask.type(torch.FloatTensor)
        mask = mask.cuda() if torch.cuda.is_available() else mask
        # mask: (BS, RAL)

        # mask the attention, then normalize
        attention = attention * mask
        F.normalize(attention, p=1, dim=1)
        attention = attention.unsqueeze(1)
        # attention: (BS, 1, RAL)

        # (BS, 1, RAL) x (BS, RAL, CS)
        context = torch.bmm(attention, val)
        context = context.squeeze(1)
        # context: (BS, CS)

        attention = attention.squeeze(1)
        # attention: (BS, RAL)

        return context, attention