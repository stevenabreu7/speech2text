import os
import yaml
import torch
import notify
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from decoder import char_to_num
from decoder import decode_train
from speller import Speller
from listener import Listener
from wsj_loader import val_loader, train_loader


class Trainer():
    def __init__(self, name, log_path, model_path, AUF, HFL, CS, VOC, HFS, EMB, lr, n, tfr, load_epochs):
        """
          This Trainer class contains everything that has to do with training 
          the LAS model and making predictions. Arguments are passed from a 
          yaml configuration file.
          Params:
            name                            name of this model
            log_path                        directory where logs are stored
            model_path                      directory where models are stored
            AUF, HFL, CS, VOC, HFS, EMB     dimensions in the network
            lr                              learning rate
            n                               number of epochs to train
            tfr                             teacher forcing rate
            load_epochs                     if pretraining, this indicates the number 
                                            of epochs for which the model was pretrained
        """
        print('\rSetting up trainer...', end='', flush=True)
        # meta parameters
        self.name = name
        self.log_path = log_path
        self.model_path = model_path

        # training parameters
        self.tfr = tfr
        self.n_epochs = n

        # gpu availability
        self.use_gpu = torch.cuda.is_available()

        # data loaders
        self.val_loader = val_loader()
        self.train_loader = train_loader()

        # networks
        self.speller = Speller(CS, VOC, HFS, EMB, tfr)
        self.listener = Listener(AUF, HFL, CS)

        # optimizer for learning
        params = [{'params': self.speller.parameters()}, {'params': self.listener.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr)

        # loss criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # move to GPU if possible
        if self.use_gpu:
            self.speller = self.speller.cuda()
            self.listener = self.listener.cuda()
            self.criterion = self.criterion.cuda()
        
        # use pretrained model if so indicated
        if load_epochs is not None:
            self.load(load_epochs)
            self.epoch_i = load_epochs
        else:
            self.epoch_i = 0
        
    def preprocess_input(self, x, y):
        """
          Preprocess the batch input.
          Params:   
            x:      data
                    [BS] (AUL*, AUF)
            y:      labels
                    [BS] (LAL*)
          Returns:
            x:      data
                    (BS, AUL, AUF)
            y:      labels
                    (BS, LAL)
            x_lens: true length of each tensor in x
                    [BS]
            y_lens: true length of each tensor in y
                    [BS]
        """
        # get the <eos> symbol code
        eos_sym = char_to_num['<eos>']

        # sorting data jointly by length of x
        z = zip(*sorted(zip(x, y), key=lambda a: a[1].shape[0], reverse=True))
        x, y = (list(l) for l in z)

        # saving lengths of x and y tensors
        x_lens = [e.size(0) for e in x]
        y_lens = [e.size(0) for e in y]

        # pad x and y
        x = rnn.pad_sequence(x, batch_first=True, padding_value=0)
        y = rnn.pad_sequence(y, batch_first=True, padding_value=eos_sym)
        # make sure x's length is divisible by eight
        if x.size(1) % 8 != 0:
            pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
            x = func.pad(x, (0, 0, 0, pad_len))
        # x: (BS, AUL, AUF) - padded
        # y: (BS, LAL)      - padded

        # turn into variables
        x = Variable(x)
        y = Variable(y, requires_grad=False)
        # x: (BS, AUL, AUF) - Variable, padded
        # y: (BS, LAL)      - Variable, padded

        # move to GPU if possible
        x = x.cuda() if self.use_gpu else x
        y = y.cuda() if self.use_gpu else y
        # x: (BS, AUL, AUF) - Variable, padded
        # y: (BS, LAL)      - Variable, padded

        return x, y, x_lens, y_lens

    def forward_batch(self, x, y, training, log_att=False):
        """
          Params:
            x:          [BS] tensors, each (L*, N), where L* varies
            y:          [BS] tensors, each (T*), where T* varies
            training:   boolean of whether or not we're training
        """
        if training:
            self.speller.train()
            self.listener.train()
        else:
            self.speller.eval()
            self.listener.eval()
        
        # preprocess the input
        x, y, x_lens, y_lens = self.preprocess_input(x, y)
        # x: (BS, AUL, AUF) - Variable, padded
        # y: (BS, LAL) - Variable, padded
        # x_lens: [BS] true length of x
        # y_lens: [BS] true length of y

        # pass x through the listener
        key, val, kv_lens = self.listener(x, x_lens)
        # key: (BS, RAL, CS)
        # val: (BS, RAL, CS)
        # kv_lens: [BS], true length of key and val

        # make the mask for the speller
        speller_mask = torch.zeros((key.size(0), key.size(1)))
        speller_mask = speller_mask.type(torch.FloatTensor)
        speller_mask = speller_mask.cuda() if self.use_gpu else speller_mask
        for batch_idx, max_time in enumerate(kv_lens):
            speller_mask[batch_idx, :max_time] = 1
        # mask: (BS, RAL)

        # get prediction from the speller
        pred = self.speller(key, val, y, speller_mask, log_att=log_att)
        pred = pred.permute(0, 2, 1)
        # pred: (BS, VOC, LAL)

        # prepare target
        y = y.data.contiguous().type(torch.LongTensor)
        y = y.cuda() if self.use_gpu else y
        # y: (BS, LAL) - padded

        # get dimensions
        BS, LAL = y.size()

        # prepare mask for loss computation
        loss_mask = torch.zeros((BS, LAL))
        loss_mask = loss_mask.type(torch.FloatTensor)
        loss_mask = loss_mask.cuda() if self.use_gpu else loss_mask
        for batch_idx, max_time in enumerate(y_lens):
            loss_mask[batch_idx, :max_time] = 1
        # mask: (BS, LAL)

        # reshape tensors for loss computation
        loss_mask = loss_mask.contiguous().view(BS*LAL)
        pred = pred.contiguous().view(BS*LAL, -1)
        y = y.contiguous().view(BS*LAL)
        # loss_mask: (BS*LAL)
        # pred: (BS*LAL, VOC)
        # y: (BS*LAL)

        # compute the loss
        loss = self.criterion(pred, y)
        # loss: (BS*LAL)

        # mask the loss
        loss = loss * loss_mask
        # loss: (BS*LAL)

        # loss for logging purposes
        log_loss = loss.sum().item() / loss_mask.sum().item()
        # log_loss: scalar

        # reconstruct original size
        loss = loss.view(BS, LAL)
        # loss: (BS, LAL)

        # sum over the loss
        loss = loss.sum(dim=1)
        # loss: (BS)

        # compute the mean over the batch
        loss = loss.mean()
        # loss: scalar

        #####################
        # 3) Backward pass

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return log_loss
    
    def generate_single(self, x, n):
        """
          Generate sentences for the given data.
          Params:
            x: tensor of size (AUL, AUF)
            n: number of random searches to make
        """
        # single batch
        x = x.unsqueeze(0)
        # x: (1, AUL, AUF)

        # make sure x's length is divisible by eight
        if x.size(1) % 8 != 0:
            pad_len = (x.size(1) // 8 + 1) * 8 - x.size(1)
            x = func.pad(x, (0, 0, 0, pad_len))
        # x: (AUL, AUF)

        if self.use_gpu:
            x = x.cuda()

        key, val, true_lens = self.listener(x, [x.size(1)])
        # key: (1, RAL, CS)
        # val: (1, RAL, CS)
        # true_lens: list with one entry

        mask = torch.zeros((1, key.size(1)))
        mask = mask.type(torch.FloatTensor)
        mask[0, :true_lens[0]] = 1
        # mask: (1, RAL)

        if self.use_gpu:
            mask = mask.cuda()

        pred = self.speller(key, val, None, mask, pred_mode=True)
        # pred: (1, LAL, VOC)
        pred = pred.squeeze(0)
        # pred: (LAL, VOC)

        p = func.softmax(pred, dim=1)
        # p: (LAL, VOC)

        # make a distribution
        distr = Categorical(p)

        # get samples
        samples = [distr.sample() for i in range(n)]
        # samples: [n] x (250)

        for idx in range(len(samples)):
            if 33 in list(samples[idx]):
                samples[idx] = samples[idx][:list(samples[idx]).index(33)+1]
        # samples: [n] x (variable_length)

        losses = []
        for idx in range(len(samples)):
            y = samples[idx]
            y = y.unsqueeze(0)
            y = y.cuda() if self.use_gpu else y
            # y: (1, YLEN)
            
            pred = self.speller(key, val, y, mask, pred_mode=True)
            pred = pred.permute(0, 2, 1)
            # pred: (1, VOC, LAL)

            y_len = y.size(1)
            pred = pred[:, :, :y_len]
            # pred: (1, VOC, YLEN)
            losses.append(self.criterion(pred, y).cpu().item())
        
        # find minimum loss
        idx = losses.index(min(losses))
        
        # sample with minimum loss
        res = samples[idx]

        # print the prediction as decoded string
        print(decode_train(res.cpu().numpy()))

        return res

    def train(self):

        while self.epoch_i < self.n_epochs:
            
            # TRAINING
            n_batches = len(self.train_loader)
            loss = 0.0

            for idx, (batch_data, batch_label) in enumerate(self.train_loader):

                la = (idx % 50 == 0)
                loss += self.forward_batch(batch_data, batch_label, training=True, log_att=la)
                cur_loss = loss / (idx+1)

                print('\r[TRAIN] Epoch {:02}  Batch {:03}/{:03}  Loss {:7.3f}  Perplexity {:7.3f}'.format(
                    self.epoch_i+1, idx+1, n_batches, cur_loss, 2**cur_loss
                ), end='', flush=True)
            print()

            notify.send(
                '[TRAIN] Epoch {:02}'.format(self.epoch_i+1), 
                'Loss {:7.3f} Perplexity {:7.3f}'.format(loss/n_batches, 2**(loss/n_batches))
            )

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

            notify.send(
                '[VAL] Epoch {:02}'.format(self.epoch_i+1), 
                'Loss {:7.3f} Perplexity {:7.3f}'.format(loss/n_batches, 2**(loss/n_batches))
            )

            # SAVE MODEL
            self.save(self.epoch_i+1)

            self.epoch_i += 1

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
    
    def load(self, epoch):
        """
          Load the model from the path specified in the config file.
          Params:
            epoch:      string to add to the end of the model path
        """
        path = self.model_path
        # get the paths from where we load the model
        speller_path = os.path.join(path, '{}_{}.speller'.format(self.name, epoch))
        listener_path = os.path.join(path, '{}_{}.listener'.format(self.name, epoch))
        # make sure the files exist
        assert os.path.exists(speller_path), 'Speller path doesnt exist'
        assert os.path.exists(listener_path), 'Listener path doesnt exist'
        # load the state dictionaries into the models
        if self.use_gpu:
            self.speller.load_state_dict(torch.load(speller_path))
            self.listener.load_state_dict(torch.load(listener_path))
        else:
            self.speller.load_state_dict(torch.load(speller_path, map_location='cpu'))
            self.listener.load_state_dict(torch.load(listener_path, map_location='cpu'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to the config file to be used")
    args = parser.parse_args()

    conf = yaml.load(open(args.config, 'r'))
    trainer = Trainer(**conf['model_params'])

    trainer.train()

    x = np.load('data/test.npy', encoding='bytes')[0]
    x = torch.Tensor(x)
    trainer.generate_single(x, 10)
