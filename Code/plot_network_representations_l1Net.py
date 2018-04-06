### Explore activations of trained networks 
#
# we focus on plotting results for L1 loss trained networks!
#
#

import torch
import argparse

import numpy as np
import operator

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
from network_DCGAN_linear import *
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
#net = torch.load('dcgan_architecture_doubleFilters_L1loss_LinearFirst_fullFilter.pth')
#net_l2 = torch.load('dcgan_architecture_doubleFilters_L2loss.pth')
#net_ce = torch.load('dcgan_architecture_doubleFilters_CrossEntLoss_threshold.pth')

# load in centering vector:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
c_vec = np.loadtxt('centeringWordVector.txt')

# load in word vector representations
model = model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ricardo/Downloads/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

def getMapping( net, tokens, thres=.05 , title_=None, saveFig=False):
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

	plotting.plot_glass_brain( '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/trial_name.nii.gz', display_mode='z', threshold=1, title=title_)

	my_arr_mask = np.ma.masked_where(bimage < thres, bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=min(0, thres))#, vmax=my_arr.max())
	if saveFig:
		plt.savefig('activation_'+title_+'.png')


# produce figures:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/presentation')
getMapping(net, ['amygdala'], thres=.2, title_='amygdala', saveFig=True) # nice!
getMapping(net, ['visual'], thres=.1, title_='visual cortex', saveFig=True)
getMapping(net, ['orbitofrontal', 'frontal'], title_='orbitofrontal cortex', saveFig=True, thres=.12)
getMapping(net, 'motor, sensory, areas, sensorimotor, primary, somatosensory, system'.split(', '), thres=.1, title_='motor cortex', saveFig=True)
getMapping(net, ['auditory'], thres=.1, title_='auditory', saveFig=True)
getMapping(net, ['memory', 'task', 'working'], title_='working memory', saveFig=True, thres=.1)
getMapping(net, ['reward'], title_='reward', thres=.15, saveFig=True)

# based on Rob/Romys suggestions
getMapping(net, 'left finger tapping'.split(' '), thres=.19, title_='left finger tapping')
getMapping(net, 'right-hand finger tapping righthand'.split(' '), thres=.19, title_='right finger tapping')

# put into 1 image
os.system('convert ')


getMapping(net, ['frontal'])
getMapping(net, ['prefrontal'])
getMapping(net, ['language'])
getMapping(net, ['cerebellum'])
getMapping(net, ['fusiform', 'gyrus'])
getMapping(net, ['precuneus'])
getMapping(net, ['thalamus'])
getMapping(net, ['dorsolateral', 'prefrontal'])
getMapping(net, ['orbitofrontal', 'cortex'])
getMapping(net, ['acc', 'cortex'])
getMapping(net, ['memory', 'task', 'working'], title_='working memory')
getMapping(net, 'memory, working, retrieval, episodic, encoding, memories'.split(', '))
getMapping(net, 'emotional, processing, neutral, emotion, arousal, valence, affective, emotionally'.split(', '))
getMapping(net, 'reward, anticipation, rewards, motivation, incentive'.split(', '))
getMapping(net, 'semantic, word, words, processing, lexical, semantically, knowledge, meaning'.split(', '))
getMapping(net, 'pain, painful, chronic, noxious, somatosensory'.split(', '))
getMapping(net, 'speech, auditory, production, perception, sounds, listening, acoustic, phonetic, syllable, syllables, stuttering, vowel, prosodic, spoken, linguistic'.split(', '))
getMapping(net, ['faces', 'social'])


# now try some word semantic relationships!
def semanticRelPrediction( tokens1, tokens2, thres=.05, norm = False):
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
	my_arr_mask = np.ma.masked_where(bimage < thres , bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1)
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())



semanticRelPrediction(['working', 'memory', 'task'], ['faces', 'neutral'])

#semanticRelPrediction( ['amygdala','emotion'], ['memory'] )
#semanticRelPrediction( ['amygdala','emotion'], ['fear'] )
#semanticRelPrediction( ['amygdala','emotion'], ['happy'] )
#semanticRelPrediction( ['amygdala','emotion'], ['sad'] )


# now we try to do reverse inference:
# we select a given part of the brain and see which word maximally 
# activates that brain "region"!
#
#
#

# first get list of all words and their vectors
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
wordVector_dict = pickle.load(open('WordVectors_reduced.p', 'rb')) # wordVector_dict = pickle.load(open('WordVectors.p', 'rb'))

def GetMostActiveWords( coords, makeFig=False ):
	"""
	Given a set of input coordinates, get the words which maximally that given voxel

	INPUT:
		coords: a list of 2D coordinates. We average the activation across all these coordinates and report words which lead to highest activation

		eg coords  = [[7,3], [13,3]]
	"""

	if makeFig:
		#plotting.plot_glass_brain( '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/trial_name.nii.gz', display_mode='z', threshold=1)

		img = plt.imread('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/clearBrain.png')
		plt.imshow(img, extent = (0, 20, 0, 20))
		plt.yticks([]); plt.xticks([]); plt.axis('off')

		for c in coords:
			plt.scatter( c[0], c[1], marker='*', s=400, edgecolor='red', facecolors='red')

		plt.tight_layout()

	#coords_mni = [((np.array(x)+100.)/10).astype('int') for x in coords]
	coords_mni = np.array( coords )
	# now we go through and check the performance of each word:
	word_performance = {}
	for word in wordVector_dict.keys():
		wvec = Variable( torch.from_numpy( wordVector_dict[word] ).float()) # convert to correct format 
		bimage = net( wvec ).data.numpy().reshape((20,20))
		word_performance[word] = bimage[::-1,:][ coords_mni[:,1], coords_mni[:,0]].mean() # - bimage.mean()

	sorted_x = sorted(word_performance.items(), key=operator.itemgetter(1))

	return sorted_x[::-1][:10]


GetMostActiveWords( coords = [[7,3], [13,3]], makeFig=True )

GetMostActiveWords( coords = [[3,9], [17,9]], makeFig=True )

GetMostActiveWords( coords = [[9,12], [11,12]], makeFig=True )

GetMostActiveWords( coords = [[7,6], [13,6]], makeFig=True )

GetMostActiveWords( coords = [[5,8], [15,8]], makeFig=True )



### now we try walking in the latent space
#
#
#

def TraverseLatentSpace( tokens1, tokens2, norm=False, stepNum=10, thres=.05 ):
	"""
	linearly traverse the word vector space starting at tokens1 and finishing at tokens2

	INPUT:
		- tokens1 : word tokens for starting representation
		- tokens2 : word tokens for final representation
		- norm    : should we take mean of tokens or not
		- stepNum : number of steps to take

	"""

	wvec_1 = getMeanVectorRepresentation( tokens1, norm=norm )
	wvec_2 = getMeanVectorRepresentation( tokens2, norm=norm )

	alpha = np.linspace(0,1, stepNum)

	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/tmp')

	for a in range(len(alpha)):
		wvec = Variable( torch.from_numpy( wvec_1 + alpha[a] * ( wvec_2 - wvec_1 ) ).float()) # convert to correct format 
		bimage = net( wvec ).data.numpy().reshape((20,20))
		my_arr_mask = np.ma.masked_where(bimage < thres , bimage)

		cmap = plt.cm.YlOrRd
		cmap.set_bad(color='white')
		limits = 100 # 90 
		plotting.plot_glass_brain( '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/trial_name.nii.gz', display_mode='z', threshold=1)
		plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
		plt.savefig( 'wordVectorTrajectory_' +str(a+1)+'.png')

	os.system('convert +append ' + ' '.join(['wordVectorTrajectory_' + str(x+1) +'.png' for x in range(len(alpha))]) + ' currentTrajectory.png')
	os.system('rm wordVectorTrajectory_*')



TraverseLatentSpace( tokens1=['working', 'memory', 'task'], tokens2=['visual'], thres=.1, stepNum=5)

TraverseLatentSpace( tokens1=['orbitofrontal', 'frontal'], tokens2=['visual'], thres=.1, stepNum=5)

TraverseLatentSpace( tokens1=['pain', 'painful'], tokens2=['auditory'], thres=.125, stepNum=5)

TraverseLatentSpace( tokens1=['pain', 'painful'], tokens2=['auditory'], thres=.125, stepNum=5)




# load in data and compare activations across L1 and L2 trained networks!
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/final')
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_testing.p' )
test_loader = DataLoader( myData, batch_size = 128 )

#myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_training.p' )
#test_loader = DataLoader( myData, batch_size = 128 )


def ComparePredictionl1Network( im_id ):
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



def ComparePredictionSeveralNetworks( im_id ):
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

	# the L2 network
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title='L2')
	wvec = Variable( torch.from_numpy( test_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage = net_l2( wvec ).data.numpy().reshape((20,20))
	my_arr_mask = np.ma.masked_where(bimage < 0.05, bimage)
	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/networkComparison')
	plt.savefig('activation_id' + str(im_id) + '_l2.png')

	# finally,the BCE network
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1, title='BCE')
	wvec = Variable( torch.from_numpy( test_loader.dataset[im_id]['wordVector'] ).float())
	wvec = wvec.resize(1,200)
	bimage = net_ce( wvec ).data.numpy().reshape((20,20))
	my_arr_mask = np.ma.masked_where(bimage < 0.05, bimage)
	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/networkComparison')
	plt.savefig('activation_id' + str(im_id) + '_BCE.png')


	os.system('convert +append activation_id'+str(im_id)+'_true.png activation_id'+str(im_id)+'_l1.png activation_id'+str(im_id)+'_l2.png activation_id'+str(im_id)+'_BCE.png res_id'+str(im_id)+'.png')
	# remove older files
	os.system('rm *_true.png')
	os.system('rm *_l1.png')
	os.system('rm *_l2.png')
	os.system('rm *_BCE.png')
	plt.close('all')