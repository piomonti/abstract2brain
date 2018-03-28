### train baseline network 
#
#
#

import torch
import argparse

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable

import os 
import cPickle as pickle 

# load in network architecture:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/networks')
from baseline_network import baseline_network

# load in data:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
#dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors.p', 'rb'))
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors.p' )

# network hyper-parameters
lr = .001
mom = .9
tr = 25
bs = 128 # batch size
D_in = 200
D_out = 20
l2_reg = .0001
hidden_dim = 400
hidden_layers = 2
verbose = True
dtype = torch.FloatTensor

# load in neurosynth dataset:


# define some variables 
x = torch.from_numpy( dat['wordVectors'][:10,  :]).float()
y = torch.from_numpy( dat['imageVectors'][:10, :]).float()

# initialize network
net = baseline_network( input_size=D_in , hidden_size=D_out*D_out, output_size=D_out, n_linear_layers=hidden_layers, n_conv_layers=0 )

# define the optimizer:
optimizer = optim.SGD( net.parameters(), lr=lr, momentum=mom, weight_decay=l2_reg )

# start training:
for e in range(tr):
	model.train() # set the model in training mode - only relevant if we are using dropout or batchnorm!
	# training for eth epoch!
	optimizer.zero_grad()
	pred = net( data )



    for batch_idx, (data, target) in enumerate( train_loader ):
      data, target = Variable( data ), Variable( target )
      # run one step of SGD
      optimizer.zero_grad() # clear all previous gradients 
      pred = model( data ) # get predictions for this batch
      loss = F.nll_loss( pred, target )
      if architecture=="van":
        # add the lagrangian cost:
        for l in model.Layers:
	  loss += lagAdjustFactor * torch.sum( torch.mul( l.lam, ( l.alpha - 1.) ) )
      loss.backward()
      optimizer.step() # take one step in direction of stochastic gradient 
      if architecture=="van":
	# we also update lagrange multipliers here - also clip alpha values to be in [0,1]!
	for l in model.Layers:
	  l.alpha.data.clamp_( max = 1 ).clamp_( min = 0 )
	  l.lam.data.add_( lr * lagAdjust * (l.alpha.data - 1. ) )
	  
      # track correct and incorrect predictions:

