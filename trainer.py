import yaml
import time
import data
import model
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

conf = yaml.load(open('config/las_config.yaml','r'))

# data loaders
val_loader = data.val_loader()
train_loader = data.train_loader()

# models
listener, speller = model.createModel(**conf['model_params'])

# learning parameters
params = [{'params': listener.parameters()}, {'params': speller.parameters()}]
l_rate = conf['training_params']['learning_rate']
n_epochs = conf['training_params']['n_epochs']
tf_rate = conf['training_params']['tf_rate']

# optimizer for learning
optimizer = torch.optim.Adam(params, l_rate)

# check GPU availability
use_gpu = torch.cuda.is_available()

# tensorboard logging
log_path = conf['meta_params']['log_folder'] + conf['meta_params']['model_name']
log_writer = SummaryWriter(log_path)

for epoch_i in range(n_epochs):

    n_batches = len(train_loader)
    epoch_start = time.time()
    batch_start = None

    # Training
    for idx, (batch_data, batch_label) in enumerate(train_loader):

        # end stop watch for previous iteration
        batch_end = time.time()
        
        if (idx+1) % 10 == 0 and idx != 0:
            print('Epoch {:02}\tBatch {:03}/{:03}\tDur {:5.3f}'.format(epoch_i, idx+1, n_batches, batch_end - batch_start), end='\r', flush=True)
        
        # start stopwatch for the next iteration
        batch_start = time.time()

        # data 
        batch_data = Variable(batch_data).type(torch.FloatTensor)
        batch_label = Variable(batch_label, requires_grad=False)

        # loss function
        criterion = nn.NLLLoss(ignore_index=0)

        # maximum length we allow for the label
        max_label_len = conf['training_params']['max_label_len']
        max_label_len = min(batch_label.size(1), max_label_len)

        # use gpu if possible
        if use_gpu:
            speller = speller.cuda()
            listener = listener.cuda()
            criterion = criterion.cuda()
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
        
        # forward pass
        optimizer.zero_grad()
        h = listener(batch_data)
        raw_pred, _ = speller(h, y=batch_label, rate=tf_rate)

        # computing prediction
        concatd = torch.cat([each_y.unsqueeze(1) for each_y in raw_pred], 1)
        pred_y = concatd[:, :max_label_len ,:].contiguous()

        # true label
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)

        # compute the loss 
        # TODO smoothening?
        assert pred_y.size() == true_y.size()
        seq_len = torch.sum(torch.sum(true_y, dim=-1), dim=-1, keepdim=True)
        loss = - torch.mean(torch.sum((torch.sum(true_y * pred_y, dim=-1) / seq_len), dim=-1))

        # TODO compute edit distance as error metric

        # backward pass
        loss.backward()
        optimizer.step()
        batch_loss = loss.cpu().data.numpy()
        
        # log this batch loss to tensorboard
        log_writer.add_scalars('loss', {'train': batch_loss}, n_batches * epoch_i + idx)
    
    print('Epoch {:02} completed in {:5.3f}s'.format(epoch_i, time.time() - epoch_start))
