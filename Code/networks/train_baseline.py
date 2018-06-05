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
from network_DCGAN_linear import * 
from data_loader import neurosynthData

# network hyper-parameters
lr = .001 / 50
mom = .9
tr = 50
bs = 128 # batch size
D_in = 200
D_out = 20
l2_reg = .001 / 10
hidden_layers = 1
verbose = True
dtype = torch.FloatTensor
networkType = 'DCGAN_linear' # should one of ['baseline', 'baseline_conv', 'DCGAN']

# load in neurosynth dataset:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/final')
#dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors.p', 'rb'))
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_training.p' )
myData_test = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_testing.p' )
train_loader = DataLoader( myData, batch_size = bs, shuffle=True )
test_loader  = DataLoader( myData_test, batch_size = bs )

# define seed
random.seed(1)

# initialize network
if networkType=='DCGAN':
	net = myDCGAN_arch()
	net.apply(weights_init)
	print(net)
elif networkType=='DCGAN_linear':
	net = myDCGAN_arch_linear()
	net.apply(weights_init)
	# set linear mapping to the identity:
	net.linearMap.weight.data.copy_(torch.eye(200))
	print(net)
elif networkType=='baseline_conv':
	net = baseline_network_conv( input_size=D_in , hidden_size=D_out*D_out, output_size=D_out, n_linear_layers=hidden_layers, n_conv_layers=0 )
else:
	net = baseline_network( input_size=D_in , hidden_size=D_out*D_out, output_size=D_out, n_linear_layers=hidden_layers )


# compute total number of parameters:
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print 'There are a total of ' + str(pytorch_total_params) + ' trainable params'

# define the optimizer:
if networkType in ['DCGAN', 'DCGAN_linear']:
	optimizer = optim.Adam(net.parameters(), lr=lr, betas=(.5, 0.999), weight_decay=l2_reg )
else:
	optimizer = optim.SGD( net.parameters(), lr=lr, momentum=mom, weight_decay=l2_reg )


loss_train = [] # track training and test loss
loss_test_track  = []

my_loss =  F.l1_loss #nn.BCEWithLogitsLoss() #  F.mse_loss # F.l1_loss

if type(my_loss).__name__ == "BCEWithLogitsLoss":
	useThres = True # we use the threshold because we are predicting binary outputs!
	thres_func = lambda x: (x>.05).float() #nn.Threshold( .05, 0 )
else:
	useThres = False 

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
		if useThres:
			loss = my_loss( pred.view(-1, D_out*D_out), thres_func( target ) )
		else:
			loss = my_loss( pred.view(-1, D_out*D_out), target )
		#loss = F.mse_loss( pred.view(-1, D_out*D_out), target )
		#loss = F.l1_loss( pred.view(-1, D_out*D_out), target )
		#loss = nn.BCEWithLogitsLoss()( pred.view(-1, D_out*D_out), target ) # thres_func( target )
		loss.backward()
		optimizer.step() # take one step in direction of stochastic gradient 
		epoch_loss += loss.data.numpy()
	# get performance on test data
	test_perf = 0
	for idx, sample in enumerate(test_loader):
		data, target = Variable( sample['wordVector'] ).float(), Variable( sample['image'] ).float()
		pred_test = net( data )
		test_perf += my_loss( pred_test.view(-1, D_out*D_out), target).data.numpy()

	# store results
	loss_train.append( epoch_loss[0] / myData.__len__() )
	loss_test_track.append( test_perf[0] / myData_test.__len__() )
	if verbose:
		print 'Epoch: ' + str(e) + '\tloss: '+ str(epoch_loss[0] / myData.__len__() ) + '\t test loss: ' + str(test_perf[0] / myData_test.__len__() )


saveModel = False
if saveModel:
	# save the model
	if networkType=='DCGAN':
		os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/models/new')
		torch.save(net, 'dcgan_architecture_doubleFilters_CrossEntLoss_threshold.pth')
	elif networkType=='DCGAN_linear':
		os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/models/new')
		torch.save(net, 'dcgan_architecture_doubleFilters_L1Loss_LinearFirst_fullFilter.pth')


# compare some images to see if it is learning anything..
if False:
	import pylab as plt; plt.ion()
	im_id = 40
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow( myData_test[im_id]['image'].reshape((20,20)))
	axarr[0].set_title('true activation')

	wvec = Variable( torch.from_numpy( test_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage_null = net( Variable(torch.from_numpy(np.zeros(200))).float().resize(1,200)).data.numpy()
	bimage = net( wvec ).data.numpy()
	axarr[1].imshow( (bimage).reshape((20,20)) )
	axarr[1].set_title('predicted activation')

	#os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/activation_plots')
	#plt.savefig('Activation_id_' + str(im_id) +'.png')


	# repeat for on training data:
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow( myData[im_id]['image'].reshape((20,20)))
	axarr[0].set_title('true activation')

	wvec = Variable( torch.from_numpy( train_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage_null = net( Variable(torch.from_numpy(np.zeros(200))).float().resize(1,200)).data.numpy()
	bimage = net( wvec ).data.numpy()
	axarr[1].imshow( (bimage).reshape((20,20)) )
	axarr[1].set_title('predicted activation')

