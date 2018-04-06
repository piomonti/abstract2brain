### we implement the same transpose convolution architecture as in the DCGAN paper
#
# this implementation is shamelessly taken from: 
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
#
# NOTE: as suggested by Aapo, we had a linear layer at first, before 
# beginning on the transpose convolutions
#


from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


nz_in = 200
nz = 200
ngf = 20
filter_size = 4*2 # originally 8
# we change dimensions slightly so that we match with image we want to have!

class myDCGAN_arch_linear( nn.Module ):
	"""
	small adjustments to DCGAN architecture 
	"""
	def __init__( self ):
		super( myDCGAN_arch_linear, self ).__init__()
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(     nz, ngf * filter_size, 5, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * filter_size),
			nn.ReLU(True),
			# current state size: (ngf*8) x 5 x 5
			nn.ConvTranspose2d( ngf * filter_size, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf ),
			nn.ReLU(True),
			# current state size: (ngf*4) x 10 x 10
			nn.ConvTranspose2d( ngf , 1, 4, 2, 1, bias=False),
			nn.Tanh()
		)

		self.linearMap = nn.Linear(nz_in, nz ) # apply linear layer first;

	def forward(self, input):
		output = self.main( self.linearMap( input ).view(-1, nz, 1, 1))
		return output
