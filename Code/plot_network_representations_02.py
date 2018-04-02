### Explore activations of trained networks 
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
from torch.utils.data import DataLoader

import os 
import cPickle as pickle 
import gensim
from nilearn import plotting
import pylab as plt; plt.ion()

# load in network architecture modules
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/networks')
from baseline_network import baseline_network
from baseline_network_conv import baseline_network_conv
from network_DCGAN import * 
from data_loader import neurosynthData

# load plotting functions:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/utils')
from plotBrains import *  # contains functions to downsample images!

def getMeanVectorRepresentation( tokens, norm=True ):
	"""
	compute the mean word embedding vector for all words in the abstract 
	"""
	abs_vec = np.zeros((200,))
	counter = 0
	for x in tokens:
		try: 
			abs_vec += (model.get_vector(x)-c_vec)
			counter += 1
		except KeyError:
			pass
	if norm:
		return abs_vec/counter
	else:
		return abs_vec

# load in pretrained networks
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/models/new')
net = torch.load('dcgan_architecture_doubleFilters_L1loss.pth')
net_l2 = torch.load('dcgan_architecture_doubleFilters_L2loss.pth')
net_ce = torch.load('dcgan_architecture_doubleFilters_CrossEntLoss.pth')

# load in centering vector:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
c_vec = np.loadtxt('centeringWordVector.txt')

# load in word vector representations
model = model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ricardo/Downloads/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

def getMapping( net, tokens, thres=.05 , title_=None):
	"""
	compute predicted brain response from text

	INPUT:
		- net    : a pretrained network used to predict images 
		- tokens : a list of words, will take mean embedding

	"""

	wvec = getMeanVectorRepresentation( tokens )
	#wvec -= c_vec
	wvec = Variable( torch.from_numpy( wvec ).float()) # convert to correct format 
	bimage = net( wvec ).data.numpy().reshape((20,20))

	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title=title_)

	my_arr_mask = np.ma.masked_where(bimage < thres, bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=min(0, thres))#, vmax=my_arr.max())


# for example:
getMapping(net, ['amygdala'], thres=.2, title_='amygdala') # nice!
getMapping(net, ['visual'], thres=.1, title_='visual cortex')
getMapping(net, ['frontal'])
getMapping(net, ['prefrontal'])
getMapping(net, ['language'])
getMapping(net, ['cerebellum'])
getMapping(net, ['fusiform', 'gyrus'])
getMapping(net, ['precuneus'])
getMapping(net, ['thalamus'])
getMapping(net, ['auditory', 'stimuli'], title_='auditory') # nice! 
getMapping(net, ['dorsolateral', 'prefrontal'])
getMapping(net, ['orbitofrontal', 'cortex'])
getMapping(net, ['acc', 'cortex'])
getMapping(net, ['memory', 'task', 'working'], title_='working memory')
getMapping(net, 'memory, working, retrieval, episodic, encoding, memories'.split(', '))
getMapping(net, 'motor, sensory, areas, sensorimotor, primary, somatosensory, system'.split(', '), thres=.1, title_='motor cortex')
getMapping(net, 'emotional, processing, neutral, emotion, arousal, valence, affective, emotionally'.split(', '))
getMapping(net, 'reward, anticipation, rewards, motivation, incentive'.split(', '))
getMapping(net, 'semantic, word, words, processing, lexical, semantically, knowledge, meaning'.split(', '))
getMapping(net, 'pain, painful, chronic, noxious, somatosensory'.split(', '))
getMapping(net, 'speech, auditory, production, perception, sounds, listening, acoustic, phonetic, syllable, syllables, stuttering, vowel, prosodic, spoken, linguistic'.split(', '))
getMapping(net, ['faces', 'social'])


# now try some word semantic relationships!
def semanticRelPrediction( tokens1, tokens2, norm = False):
	"""
	we add the values for tokens1 and subtract the mean for tokens2! 

	INPUT:
		- tokens1 : word tokens to add
		- tokens2 : word tokens to subtract
		- norm    : should we take mean of tokens or not
	"""
	wvec_1 = getMeanVectorRepresentation( tokens1, norm=norm )
	wvec_2 = getMeanVectorRepresentation( tokens2, norm=norm )

	wvec = Variable( torch.from_numpy( wvec_1 - wvec_2 ).float()) # convert to correct format 
	bimage = net( wvec ).data.numpy().reshape((20,20))
	my_arr_mask = np.ma.masked_where(bimage < 0.05, bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1)
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())



semanticRelPrediction( ['amygdala','emotion'], ['memory'] )

semanticRelPrediction( ['amygdala','emotion'], ['fear'] )

semanticRelPrediction( ['amygdala','emotion'], ['happy'] )

semanticRelPrediction( ['amygdala','emotion'], ['sad'] )

# load in data and compare activations across L1 and L2 trained networks!
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_testing.p' )
test_loader = DataLoader( myData, batch_size = 128 )


def ComparePredictionNetworks( im_id ):
	"""
	compare predictions for a given abstract 
	INPUT:
		- im_id: abstract id (starting at 0, ending at 8096)
	"""

	# plot true activation first:
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title='True')
	bimage_true = myData[im_id]['image'].reshape((20,20))
	my_arr_mask = np.ma.masked_where(bimage_true < 0.05, bimage_true)
	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/networkComparison')
	plt.savefig('activation_id' + str(im_id) + '_true.png')

	# plot l1 network activation next:
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title='L1')
	wvec = Variable( torch.from_numpy( test_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage = net( wvec ).data.numpy().reshape((20,20))
	my_arr_mask = np.ma.masked_where(bimage < 0.05, bimage)
	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/networkComparison')
	plt.savefig('activation_id' + str(im_id) + '_l1.png')

	# finally, the L2 network
	#os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	#plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title='L2')
	#wvec = Variable( torch.from_numpy( test_loader.dataset[im_id]['wordVector'] ).float())
	#wvec = wvec.resize(1,200)
	#bimage = net_l2( wvec ).data.numpy().reshape((20,20))
	#my_arr_mask = np.ma.masked_where(bimage < 0.05, bimage)
	#cmap = plt.cm.YlOrRd
	#cmap.set_bad(color='white')
	#limits = 100 # 90 
	#plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	#os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/networkComparison')
	#plt.savefig('activation_id' + str(im_id) + '_l2.png')

	#os.system('convert +append activation_id'+str(im_id)+'_true.png activation_id'+str(im_id)+'_l1.png activation_id'+str(im_id)+'_l2.png res_id'+str(im_id)+'.png')
	os.system('convert +append activation_id'+str(im_id)+'_true.png activation_id'+str(im_id)+'_l1.png PredictionComparison_id'+str(im_id)+'.png')
	# remove older files
	os.system('rm *_true.png')
	os.system('rm *_l1.png')
	#os.system('rm *_l2.png')
