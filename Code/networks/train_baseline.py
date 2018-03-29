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
from torch.utils.data import DataLoader

import os 
import cPickle as pickle 

# load in network architecture:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/networks')
from baseline_network import baseline_network
from baseline_network_conv import baseline_network_conv
from network_DCGAN import * 
from data_loader import neurosynthData

# network hyper-parameters
lr = .001
mom = .9
tr = 100
bs = 128 # batch size
D_in = 200
D_out = 20
l2_reg = .00001
hidden_layers = 1
verbose = True
dtype = torch.FloatTensor
networkType = 'DCGAN' # should one of ['baseline', 'baseline_conv', 'DCGAN']

# load in neurosynth dataset:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
#dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors.p', 'rb'))
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_SCALED.p' )
train_loader = DataLoader( myData, batch_size = bs )

# define seed
random.seed(1)

# initialize network
if networkType=='DCGAN':
	net = myDCGAN_arch()
	net.apply(weights_init)
	print(net)
elif networkType=='baseline_conv':
	net = baseline_network_conv( input_size=D_in , hidden_size=D_out*D_out, output_size=D_out, n_linear_layers=hidden_layers, n_conv_layers=0 )
else:
	net = baseline_network( input_size=D_in , hidden_size=D_out*D_out, output_size=D_out, n_linear_layers=hidden_layers )

# define the optimizer:
if networkType=='DCGAN':
	optimizer = optim.Adam(net.parameters(), lr=lr, betas=(.5, 0.999))
else:
	optimizer = optim.SGD( net.parameters(), lr=lr, momentum=mom, weight_decay=l2_reg )

thres_func = nn.Threshold( .05, 0 )
# start training:
for e in range(tr):
	net.train() # set the model in training mode - only relevant if we are using dropout or batchnorm!
	# training for eth epoch!
	optimizer.zero_grad()
	epoch_loss = 0
	for batch_idx, sample in enumerate( train_loader ):
		data, target = Variable( sample['wordVector'] ).float(), Variable( sample['image'] ).float()
		# run one step of SGD
		optimizer.zero_grad() # clear all previous gradients 
		pred = net( data ) # get predictions for this batch
		loss = F.l1_loss( pred.view(-1, D_out*D_out), target )
		#loss = nn.BCEWithLogitsLoss()( pred.view(-1, D_out*D_out), target ) # thres_func( target )
		loss.backward()
		optimizer.step() # take one step in direction of stochastic gradient 
		epoch_loss += loss.data.numpy()
	if verbose:
		print 'Epoch: ' + str(e) + '\tloss: '+ str(epoch_loss)




# compare some images to see if it is learning anything at all..
if False:
	import pylab as plt; plt.ion()
	im_id = 40
	f, axarr = plt.subplots(1,3)
	axarr[0].imshow( myData[im_id]['image'].reshape((20,20)))

	wvec = Variable( torch.from_numpy( train_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage_null = net( Variable(torch.from_numpy(np.zeros(200))).float().resize(1,200)).data.numpy()
	bimage = net( wvec ).data.numpy()
	axarr[1].imshow( (bimage).reshape((20,20)) )
	axarr[2].imshow( (bimage-bimage_null).reshape((20,20)) )