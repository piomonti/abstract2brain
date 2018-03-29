### Simple baseline network - no (de)convolutions used
#
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
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/networks')
from network_modules import MLP, View 


class baseline_network_conv( nn.Module ):
	"""
	a simple feedforward network from mean word vector representation to
	2D brain activation 

	NOTE: we do note employ deconvolutions, just reshape the output and 
	(possibly) apply some convolutions.

	This network implements a function from \textbf{R}^d -> \textbf{R}^k \times \textbf{R}^k
	where d is the dimension of word embeddings and k is 2D image dimension

	"""

	def __init__( self, input_size, hidden_size, output_size, n_linear_layers, n_conv_layers ):
		"""
		INPUT:
			- input_size      : dimension of word vector embeddings
			- hidden_size     : dimension of linear hidden layers (try to match with dimension of output images)
			- output_size     : size output image (e.g., if 20, then image will be 20 by 20)
			- n_linear_layers : number of fully connected layers
			- n_conv_layers   : number of convolution layers

		"""
		super( baseline_network_conv, self ).__init__()

		self.linear1st = nn.Linear( input_size, hidden_size ) # map from data dim to dimension of hidden units
		self.FC_layers = nn.ModuleList( [ MLP( hidden_size ) for i in range(n_linear_layers ) ] )
		# add final linear step to get right output_size
		#self.linearResizeLayer = nn.Linear( hidden_size, output_size*output_size )
		# now reshape output  
		self.myreshape = View( output_size )
		# apply convolution layers
		#self.conv_layer1 = nn.Sequential( nn.ConvTranspose2d(1, 1, kernel_size=5, padding=2)) # ,nn.BatchNorm2d(1)) #,nn.ReLU() )
		self.conv_layer1 = nn.Sequential( nn.Conv2d(1, 1, kernel_size=5, padding=2)) # ,nn.BatchNorm2d(1) )#,nn.ReLU() )
		#self.conv_layer2 = nn.Sequential( nn.Conv2d(1, 1, kernel_size=5, padding=2)) # ,nn.BatchNorm2d(1) )#,nn.ReLU() )
		#self.conv_layer2 = nn.Sequential( nn.ConvTranspose2d(1, 1, kernel_size=5, padding=2))# ,nn.BatchNorm2d(1),nn.ReLU() )
		

	def forward( self, x ):
		"""
		forward pass through the network
		"""
		x = self.linear1st( x )
		for current_layer in self.FC_layers:
			x = current_layer( x )
		#x = self.linearResizeLayer( x )
		x = self.myreshape( x )
		x = self.conv_layer1( x )
		#x = self.conv_layer2( x )
		return  x 
