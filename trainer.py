import os
import yaml
import time
import data
import model
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

class Trainer():
    def __init__(self, config_path, logging, debug=False):
        """
          Trainer class to train the LAS network.
          Configurations are read from the given file.
        """
        # load configurations
        self.conf = yaml.load(open(config_path, 'r'))
        self.name = self.conf['meta_params']['model_name']

        # data loaders
        self.val_loader = data.val_loader()
        self.train_loader = data.train_loader()

        # models
        self.listener, self.speller = model.createModel(**self.conf['model_params'])

        # learning parameters
        params = [{'params': self.listener.parameters()}, {'params': self.speller.parameters()}]
        l_rate = self.conf['training_params']['learning_rate']
        self.tf_rate = self.conf['training_params']['tf_rate']
        self.n_epochs = self.conf['training_params']['n_epochs']

        # optimizer for learning
        self.optimizer = torch.optim.Adam(params, l_rate)

        # check GPU availability
        self.use_gpu = torch.cuda.is_available()

        # move networks to GPU if possible
        if self.use_gpu:
            self.speller = self.speller.cuda()
            self.listener = self.listener.cuda()
        
        # whether or not to log and debug
        self.logging = logging
        self.debug = debug

        # tensorboard logging
        log_path = self.conf['meta_params']['log_folder']
        log_path += self.name
        self.log_writer = SummaryWriter(log_path)
    
    def train_batch(self, batch_data, batch_label):
        """
          Forward and backward pass for one batch.
          Params:
            batch_data:     tensor of size B x L x N
            batch_label:    tensor of size B x T
            debug:          whether or not to display debugging messages
          Returns:
            batch_loss:     loss for this batch
        """
        # data 
        batch_data = Variable(batch_data).type(torch.FloatTensor)
        batch_label = Variable(batch_label, requires_grad=False)

        # loss function
        criterion = nn.NLLLoss(ignore_index=0)

        # maximum length we allow for the label
        max_label_len = self.conf['training_params']['max_label_len']
        max_label_len = min(batch_label.size(1), max_label_len)

        # use gpu if possible
        if self.use_gpu:
            criterion = criterion.cuda()
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
        
        # forward pass
        self.optimizer.zero_grad()
        h = self.listener(batch_data)
        raw_pred, _ = self.speller(h, y=batch_label, rate=self.tf_rate)

        # computing prediction
        # raw_pred is a list of tensors of size B x C
        # turning it into a list of tensors of B x 1 x C, then 
        # into a tensor of size B x T x C
        concatd = torch.cat([each_y.unsqueeze(1) for each_y in raw_pred], 1)
        # cutting off the max length of the sequences
        pred_y = concatd[:, :max_label_len ,:].contiguous()
        # permute pred_y from B x T x C to B x C x T
        pred_y = pred_y.permute(0, 2, 1)

        # true label
        true_y = batch_label[:, :max_label_len].contiguous()
        true_y = true_y.type(torch.cuda.LongTensor) if self.use_gpu else true_y.type(torch.LongTensor)

        if self.debug:
            print(true_y.size())
            print(pred_y.size())

        # compute the loss 
        # TODO smoothening?
        # assert pred_y.size() == true_y.size()
        # seq_len = torch.sum(torch.sum(true_y, dim=-1), dim=-1, keepdim=True)
        # loss = - torch.mean(torch.sum((torch.sum(true_y * pred_y, dim=-1) / seq_len), dim=-1))
        loss = criterion(pred_y, true_y)

        # TODO compute edit distance as error metric

        # backward pass
        loss.backward()
        self.optimizer.step()
        batch_loss = loss.cpu().data.numpy()

        return batch_loss
    
    def train(self):
        """
          Train the listener and speller simultaneously. 
          All hyperparameters are specified in the configuration file
          that was passed into the trainer class.
        """

        for epoch_i in range(self.n_epochs):

            n_batches = len(self.train_loader)
            epoch_start = time.time()
            batch_start = None

            # Training
            for idx, (batch_data, batch_label) in enumerate(self.train_loader):
                
                if self.debug:
                    print(batch_data.size())
                    print(batch_label.size())

                # start stopwatch for the next iteration
                batch_start = time.time()

                # forward and backward pass for this batch
                batch_loss = self.train_batch(batch_data, batch_label, debug)

                # end stop watch for iteration
                batch_end = time.time()
                
                # if (idx+1) % 10 == 0 and idx != 0:
                if self.logging:
                    print('Epoch {:02}\tBatch {:03}/{:03}\tLoss {:7.3f}\tDur {:5.3f}'.format(epoch_i+1, idx+1, n_batches, batch_loss, batch_end - batch_start), end='\r', flush=True)

                # log this batch loss to tensorboard
                self.log_writer.add_scalars('loss', {'train': batch_loss}, n_batches * epoch_i + idx)
            
            if self.logging:
                print('Epoch {:02} completed in {:5.3f}s'.format(epoch_i, time.time() - epoch_start))
            
            # save the current model
            self.save()
    
    def save(self, add):
        """
          Save the current model to the path specified in the config file.
          Params:
            add:    string to add to the end of the model path
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

trainer = Trainer('config/las_config.yaml', logging=True, debug=False)
trainer.train()