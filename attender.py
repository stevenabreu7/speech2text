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
        # ATTENTION STUFF
        self._attention = []

    def forward(self, key, val, query, mask):
        """
          Parameters:
            key:    key output by listener, tensor of size (BS, RAL, CS)
            val:    value output by listener, tensor of size (BS, RAL, CS)
            query:  this is the decoder output, tensor of size (BS, CS)
            mask:   tensor for the mask (true lengths) of size (BS, RAL)
          Output:
            c:      context tensor of size (BS, CS)
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
        # ATTENTION STUFF
        self._attention.append(attention[0].clone().detach().cpu()))

        # make sure mask is correct type
        mask = mask.type(torch.FloatTensor)
        mask = mask.cuda() if torch.cuda.is_available() else mask

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

