import os
import yaml
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as func
from torch.autograd import Variable
from speller import Speller
from listener import Listener
from decoder import decode_train
from wsj_loader import val_loader, train_loader


EOS = 33


class Trainer():
    def __init__(self, name, log_path, model_path, AUF, HFL, CS, VOC, HFS, EMB, lr, n, tfr, load_model, load_epochs):
        print('\rSetting up trainer...', end='', flush=True)
        self.name = name
        self.log_path = log_path
        self.model_path = model_path

        if load_model is not None:
            self.load(load_model, load_epochs)
            self.epoch_i = load_epochs
        else:
            self.epoch_i = 0

        self.tfr = tfr
        self.n_epochs = n

        self.use_gpu = torch.cuda.is_available()

        self.val_loader = val_loader()
        self.train_loader = train_loader()

        self.speller = Speller(CS, VOC, HFS, EMB, tfr)
        self.listener = Listener(AUF, HFL, CS)

        params = [{'params': self.speller.parameters()}, {'params': self.listener.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr)

        self.criterion = nn.CrossEntropyLoss()

        if self.use_gpu:
            self.speller = self.speller.cuda()
            self.listener = self.listener.cuda()
            self.criterion = self.criterion.cuda()

    def forward_batch(self, x, y, training):
        """
          Params:
            x:  list of BS tensors, each of size L* x N, where L* varies
            y:  list of BS tensors, each of size T*, where T* varies
            training:
                boolean of whether or not we're training (for backward pass)
        """
        batch_size = len(x)
        #####################
        # 1) Preprocessing

        # sorting data jointly by length of x
        z = zip(*sorted(zip(x, y), key=lambda a: a[1].shape[0], reverse=True))
        x, y = (list(l) for l in z)

        # saving length of x and y
        x_lens = [e.size(0) for e in x]
        y_lens = [e.size(0) for e in y]

        # pad x and y
        x = rnn.pad_sequence(x, batch_first=True, padding_value=0)
        y = rnn.pad_sequence(y, batch_first=True, padding_value=EOS)
        # make sure x's length is divisible by eight
        if x.size(1) % 8 != 0:
            pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
            x = func.pad(x, (0, 0, 0, pad_len))
        # x: (BS, AUL, AUF)
        # y: (BS, LAL)

        #####################
        # 2) Forward pass

        # prepare x and y
        x = Variable(x)
        y = Variable(y, requires_grad=False)
        x = x.cuda() if self.use_gpu else x
        y = y.cuda() if self.use_gpu else y
        # x: (BS, AUL, AUF) - Variable, padded
        # y: (BS, LAL)      - Variable, padded

        # zero gradients
        self.optimizer.zero_grad()

        # pass x through the listener
        key, val, true_lens = self.listener(x, x_lens)
        # key: (BS, RAL, CS)
        # val: (BS, RAL, CS)
        # true_lens: [BS], true length of key and val

        # make the mask
        mask = torch.zeros((key.size(0), key.size(1)))
        mask = mask.type(torch.FloatTensor)
        mask = mask.cuda() if self.use_gpu else mask
        for batch_idx, max_time in enumerate(true_lens):
            mask[batch_idx, :max_time] = 1
        # mask: (BS, RAL)

        # get prediction from the speller
        pred = self.speller(key, val, y, mask)
        pred = pred.permute(0, 2, 1)
        # pred: (BS, VOC, LAL)

        # prepare real prediction
        y = y.data.contiguous().type(torch.LongTensor)
        y = y.cuda() if self.use_gpu else y
        # y: (BS, LAL)  - packed, padded

        # compute the loss
        loss = 0.0
        for idx in range(batch_size):
            t = y[idx, :y_lens[idx]]
            t = t.unsqueeze(0)
            p = pred[idx, :, :y_lens[idx]]
            p = p.unsqueeze(0)
            loss += self.criterion(p, t) / float(batch_size)

        #####################
        # 3) Backward pass

        if training:
            loss.backward()
            self.optimizer.step()

        return loss.cpu().item()
    
    def train(self):

        while self.epoch_i < self.n_epochs:
            
            # TRAINING
            n_batches = len(self.train_loader)
            loss = 0.0

            for idx, (batch_data, batch_label) in enumerate(self.train_loader):

                loss += self.forward_batch(batch_data, batch_label, training=True)
                cur_loss = loss / (idx+1)

                print('\r[TRAIN] Epoch {:02}  Batch {:03}/{:03}  Loss {:7.3f}  Perplexity {:7.3f}'.format(
                    self.epoch_i+1, idx+1, n_batches, cur_loss, 2**cur_loss
                ), end='', flush=True)
            print()

            # VALIDATION
            n_batches = len(self.val_loader)
            loss = 0.0

            for idx, (batch_data, batch_label) in enumerate(self.val_loader):

                loss += self.forward_batch(batch_data, batch_label, training=False)
                cur_loss = loss / (idx+1)

                print('\r[VAL] Epoch {:02}  Batch {:03}/{:03}  Loss {:7.3f}  Perplexity {:7.3f}'.format(
                    self.epoch_i+1, idx+1, n_batches, cur_loss, 2**cur_loss
                ), end='', flush=True)
            print()

            # SAVE
            self.save(self.epoch_i+1)
    
    def save(self, epoch):
        """
          Save the current model to the path specified in the config file.
          Params:
            epoch:    number to add to the end of the model path
        """
        # create folder if it doesn't exist yet
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        # get the paths to which we save the model
        speller_path = os.path.join(path, '{}_{}.speller'.format(self.name, epoch))
        listener_path = os.path.join(path, '{}_{}.listener'.format(self.name, epoch))
        # save the state dictionaries
        torch.save(self.speller.state_dict(), speller_path)
        torch.save(self.listener.state_dict(), listener_path)
    
    def load(self, model_name, epoch):
        """
          Load the model from the path specified in the config file.
          Params:
            model_name: name of the model to load
            epoch:      string to add to the end of the model path
        """
        path = self.model_path
        # get the paths from where we load the model
        speller_path = os.path.join(path, '{}_{}.speller'.format(model_name, epoch))
        listener_path = os.path.join(path, '{}_{}.listener'.format(model_name, epoch))
        # make sure the files exist
        assert os.path.exists(speller_path), 'Speller path doesnt exist'
        assert os.path.exists(listener_path), 'Listener path doesnt exist'
        # load the state dictionaries into the models
        self.speller.load_state_dict(torch.load(speller_path))
        self.listener.load_state_dict(torch.load(listener_path))

conf = yaml.load(open('config/las_config.yaml', 'r'))
trainer = Trainer(**conf['model_params'])
trainer.train()