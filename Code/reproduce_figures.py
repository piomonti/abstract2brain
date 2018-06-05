### reproduce all figures from paper
#
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

## DEFINE SOME LOCATIONS:
brainMapLoc = '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/'
dataLoc = '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/'
figureLoc = '/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/MIDL'

# ------------------------------------------------------------------------------------
############ 1) show the effect of downsampling on images!
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/')
res = pickle.load(open('Abstract_MNI_raw.p', 'rb'))
kernelSize = 10
downsampleSize = 10

os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/MIDL')

def plotDifferenceInDownSample( pid, saveFig = False ):
	"""
	we plot the original image and the downsampled image (which is used to train the DNN)

	INPUT:
		- pid: publication ID
		- saveFig: should we save the figure
	"""

	orig_img = Get_2d_smoothed_activation( np.array(res[res.keys()[pid]]['MNI']), kernelSize)
	down_samp = downsample2d( orig_img, downsampleSize)

	# plot original:
	plotting.plot_glass_brain( brainMapLoc + 'trial_name.nii.gz', display_mode='z', threshold=1, title=title_)
	my_arr_mask = np.ma.masked_where(orig_img < .001, orig_img)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=min(0, thres))#, vmax=my_arr.max())

	if saveFig:
		os.chdir(figureLoc)
		plt.savefig('orig_img_'+str(pid)+'.png')

	# plot downsampled:
	plotting.plot_glass_brain( brainMapLoc + 'trial_name.nii.gz', display_mode='z', threshold=1, title=title_)
	my_arr_mask = np.ma.masked_where(down_samp < .001, down_samp)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=min(0, thres))#, vmax=my_arr.max())

	if saveFig:
		os.chdir(figureLoc)
		plt.savefig('downsamp_img_'+str(pid)+'.png')

		# combine both figures:
		os.system('convert +append orig_img_'+str(pid)+'.png downsamp_img_'+str(pid)+'.png preproc_pid_'+str(pid)+'.png')
		os.system('rm orig_img_'+str(pid)+'.png')
		os.system('rm downsamp_img_'+str(pid)+'.png')


for x in [50, 100, 1000]:
	plotDifferenceInDownSample( x , saveFig=True )

# convert into a single image
os.system('convert -append preproc_pid_* downsampling_image.png')


# ------------------------------------------------------------------------------------
############ 2) plot some predicted brain regions for specific word inputs!

def getMeanVectorRepresentation( tokens, norm=True, center=True ):
	"""
	compute the mean word embedding vector for all words in the abstract 
	"""
	abs_vec = np.zeros((200,))
	counter = 0
	for x in tokens:
		try: 
			if center:
				abs_vec += (model.get_vector(x)-c_vec)
			else:
				abs_vec += (model.get_vector(x))
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

# load in centering vector:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
c_vec = np.loadtxt('centeringWordVector.txt')

# load in word vector representations
model = model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ricardo/Downloads/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

def getMapping( net, tokens, thres=.05 , center=True, title_=None, saveFig=False, upsample=False):
	"""
	compute predicted brain response from text

	INPUT:
		- net    : a pretrained network used to predict images 
		- tokens : a list of words, will take mean embedding

	"""

	wvec = getMeanVectorRepresentation( tokens, center=center )
	#wvec -= c_vec
	wvec = Variable( torch.from_numpy( wvec ).float()) # convert to correct format 
	bimage = net( wvec ).data.numpy().reshape((20,20))

	plotting.plot_glass_brain( '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/trial_name.nii.gz', display_mode='z', threshold=1, title=title_)

	if upsample:
		bimage = upsampleImage( bimage, 10 )
	my_arr_mask = np.ma.masked_where(bimage < thres, bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=min(0, thres))#, vmax=my_arr.max())
	if saveFig:
		plt.savefig('activation_'+title_+'.png')


os.chdir(figureLoc)
getMapping(net, ['amygdala'], thres=.22, title_='amygdala', upsample=True, saveFig=True)
getMapping(net, ['visual'], thres=.1, title_='visual cortex', saveFig=True, upsample=True)
getMapping(net, ['orbitofrontal', 'frontal'], title_='orbitofrontal cortex', saveFig=True, upsample=True, thres=.12)
getMapping(net, 'motor, sensory, areas, sensorimotor, primary, somatosensory, system'.split(', '), thres=.1, title_='motor cortex', saveFig=True, upsample=True)
getMapping(net, ['auditory'], thres=.1, title_='auditory', saveFig=True, upsample=True)
getMapping(net, ['memory', 'task', 'working'], title_='working memory', saveFig=True, thres=.1, upsample=True)
getMapping(net, 'finger tapping'.split(' '), upsample=True, thres=.125, saveFig=True, title_='Finger tapping')


# combine into a single figure:
os.system('convert +append activation_visual\ cortex.png activation_motor\ cortex.png activation_orbitofrontal\ cortex.png topRow_forward.png')
os.system('convert +append activation_working\ memory.png activation_auditory.png activation_amygdala.png bottomRow_forward.png')

os.system('convert -append topRow_forward.png bottomRow_forward.png Forward_image.png')

# also the finger tapping example:
getMapping(net, 'left finger tapping'.split(' '), thres=.16, title_='left finger tapping', upsample=True)
getMapping(net, 'right finger tapping right-hand right right right right right'.split(' '), thres=.16, title_='right finger tapping', upsample=True)

# another nice one:
getMapping(net, 'face faces'.split(' '), upsample=True)

# another nice one:
getMapping(net, ['language', 'comprehension'], upsample=True, thres=.13, title_='language comprehension')


# ------------------------------------------------------------------------------------
############ 3) do reverse inference: find words which lead to maximum activation in particular voxels

# first get list of all words and their vectors
os.chdir(dataLoc)
wordVector_dict = pickle.load(open('WordVectors_reduced.p', 'rb')) # wordVector_dict = pickle.load(open('WordVectors.p', 'rb'))

def GetMostActiveWords( coords, saveFig=False, title_='' ):
	"""
	Given a set of input coordinates, get the words which maximally that given voxel

	INPUT:
		coords: a list of 2D coordinates. We average the activation across all these coordinates and report words which lead to highest activation

		eg coords  = [[7,3], [13,3]]
	"""

	#plotting.plot_glass_brain( '/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/trial_name.nii.gz', display_mode='z', threshold=1)

	img = plt.imread('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/clearBrain.png')
	plt.imshow(img, extent = (0, 20, 0, 20))
	plt.yticks([]); plt.xticks([]); plt.axis('off')

	for c in coords:
		plt.scatter( c[0], c[1], marker='*', s=600, edgecolor='red', facecolors='red')

	plt.tight_layout()
	if saveFig:
		plt.savefig('reverse_'+title_+'.png')


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


os.chdir(figureLoc)
GetMostActiveWords( coords = [[7,3], [13,3]], saveFig=True , title_='visual')

GetMostActiveWords( coords = [[3,9], [17,9]], saveFig=True , title_='auditory')

GetMostActiveWords( coords = [[9,12], [11,12]], saveFig=True, title_='reward' )

GetMostActiveWords( coords = [[7,9], [13,9]], saveFig=True, title_='motor' )

GetMostActiveWords( coords = [[7,6], [13,6]], saveFig=False )

GetMostActiveWords( coords = [[8,18], [12,18]], saveFig=False )


# ------------------------------------------------------------------------------------
############ 4) study some semantic relationships by doing vector addition/subtraction!

# now try some word semantic relationships!
def semanticRelPrediction( tokens1, tokens2, thres=.05, norm = False, upsample=False, plotBrain=True, saveFig=True):
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
	if upsample:
		bimage = upsampleImage( bimage, 10 )

	my_arr_mask = np.ma.masked_where(bimage < thres , bimage)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	if plotBrain:
		plotting.plot_glass_brain( brainMapLoc+'trial_name.nii.gz', display_mode='z', threshold=1)
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())

	if saveFig:
		plt.savefig('activationSemantics_'+'_'.join(tokens1)+ '_Sub_' +'_'.join(tokens2) + '.png')

os.chdir(figureLoc)
#semanticRelPrediction(['working', 'memory', 'task'], ['attention'], norm=False, thres=.1)

semanticRelPrediction([ 'faces'], ['vision'], norm=False, thres=.13, upsample=True, plotBrain=True)

semanticRelPrediction([ 'motor'], ['finger', 'tapping'], norm=False, thres=.13, upsample=True, plotBrain=True)

semanticRelPrediction(['working', 'memory'], ['executive'], norm=False, thres=.125, upsample=True, plotBrain=True)

semanticRelPrediction(['working', 'memory'], ['visual'], norm=False, thres=.13, upsample=True, plotBrain=True)

semanticRelPrediction(['decision', 'task'], ['executive'], norm=False, thres=.13, upsample=True, plotBrain=True)

semanticRelPrediction(['language', 'comprehension'], ['listen'], norm=False, thres=.14, upsample=True, plotBrain=False)


def PlotAllSubPlotsSemantic( tokens1, tokens2, thres=.05, norm = False, upsample=True, title1='', title2=''):
	"""
	plot brain maps for both tokens separately and then the difference as well!
	"""

	# plot for tokens 1
	getMapping(net, tokens1, upsample=True, title_=title1, saveFig=True, thres=thres[0])

	# plot for tokens 2
	getMapping(net, tokens2, upsample=True, title_=title2, saveFig=True, thres=thres[1])

	# finally take the difference:
	semanticRelPrediction(tokens1, tokens2, norm=False, thres=thres[2], upsample=True, plotBrain=True)

	# combine all together
	os.system('convert +append activation_'+title1+'.png ' + 'activation_'+title2+'.png ' + 'activationSemantics_'+'_'.join(tokens1)+ '_Sub_' +'_'.join(tokens2) + '.png ' + 'FullactivationSemantics_'+title1+ '_Sub_' +title2 + '.png')

PlotAllSubPlotsSemantic(['working', 'memory','task'], ['executive'], norm=True, thres=[.1,.1, .13], upsample=True, title1='working_memory', title2='executive')

PlotAllSubPlotsSemantic(['language', 'comprehension'], ['listen'], norm=False, thres=[.1, .1,.14], upsample=True, title1='language_comp.', title2='listening')

PlotAllSubPlotsSemantic(['self','referential'], 'finger tapping tap'.split(' '), thres=[.075, .1, .075], upsample=True, title1='self_referential', title2='finger_tapping')

# ------------------------------------------------------------------------------------
############ 5) we can also traverse the latent space as done in the DCGAN paper


def TraverseLatentSpace( tokens1, tokens2, norm=False, stepNum=10, thres=.05, upsample=True, title1='', title2='' ):
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

	for a in range(len(alpha)):
		wvec = Variable( torch.from_numpy( wvec_1 + alpha[a] * ( wvec_2 - wvec_1 ) ).float()) # convert to correct format 
		bimage = net( wvec ).data.numpy().reshape((20,20))
		if upsample:
			bimage = upsampleImage( bimage, 10 )

		my_arr_mask = np.ma.masked_where(bimage < thres , bimage)

		cmap = plt.cm.YlOrRd
		cmap.set_bad(color='white')
		limits = 100 # 90 
		if a==0:
			plotting.plot_glass_brain( brainMapLoc+'trial_name.nii.gz', display_mode='z', threshold=1, title=title1)
		elif a==(len(alpha)-1):
			plotting.plot_glass_brain( brainMapLoc+'trial_name.nii.gz', display_mode='z', threshold=1, title=title2)
		else:
			plotting.plot_glass_brain( brainMapLoc+'trial_name.nii.gz', display_mode='z', threshold=1)


		plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
		plt.savefig( 'wordVectorTrajectory_' +str(a+1)+'.png')

	os.system('convert +append ' + ' '.join(['wordVectorTrajectory_' + str(x+1) +'.png' for x in range(len(alpha))]) + ' currentTrajectory.png')
	os.system('rm wordVectorTrajectory_*')


os.chdir(figureLoc)
TraverseLatentSpace( tokens1=['working', 'memory', 'task'], tokens2=['visual'], thres=.1, stepNum=5, title1='working memory', title2='visual cortex')

TraverseLatentSpace( tokens1=['orbitofrontal', 'frontal'], tokens2=['visual'], thres=.1, stepNum=5, title1='orbitofrontal cortex', title2='visual cortex')

TraverseLatentSpace( tokens1=['language', 'comprehension'], tokens2=['auditory'], thres=.1, stepNum=5, title1='language comprehension', title2='auditory')

TraverseLatentSpace( tokens1=['language', 'comprehension'], tokens2=['listening'], thres=.1, stepNum=5, title1='language comprehension', title2='listening')

TraverseLatentSpace( tokens1=['pain', 'painful'], tokens2=['auditory'], thres=.125, stepNum=5)

# left to right finger tapping
TraverseLatentSpace( tokens1='left finger tapping'.split(' '), tokens2= 'right finger tapping right-hand right right right right right'.split(' '), thres=.16, stepNum=5,norm=True)



# ------------------------------------------------------------------------------------
############ 6) Learn a new vector representation for words based on our features!
#
# this is following Aapo's suggestion :) 
#

# load in data
os.chdir(dataLoc)
os.chdir('final')
dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors_training.p', 'rb'))

# put word vectors into a matrix
word_mat = np.zeros(( len(wordVector_dict.keys()), 200))
for k in range(len(wordVector_dict.keys())):
	word_mat[k,:] = wordVector_dict[ wordVector_dict.keys()[k] ]

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from sklearn.preprocessing import scale
word_mat = scale(word_mat)
pdist_orig = cosine_distances( word_mat ) #pairwise_distances( word_mat )

# project all word vectors into the final representation
word_rep_dnn = Variable( torch.from_numpy( word_mat ).float())

def GetFinalRepresentation(net, vectors):
	"""
	Get the final hidden representation
	"""

	n = vectors.size()[0]
	net_steps = net.children().next()
	out = vectors.view(-1, 200, 1, 1)
	for i in range(len(net_steps)-2):
		out = net_steps[i](out)
		print out.size()

	out = out.view(n, -1).data.numpy()

	return out

def GetBrainMap(net, vectors):
	"""
	Get the predicted brain map for each word!
	"""

	n = vectors.size()[0]
	out = net( vectors )
	out = out.view(n, -1).data.numpy()

	return out


our_rep = GetFinalRepresentation( net, word_rep_dnn ) # actually I need to do this on words!
our_rep = np.array(our_rep)

# run PCA on this representation and project into 200 dims
pca_mod = PCA(n_components=200)
pca_mod.fit( our_rep )
word_mat_dnn = pca_mod.transform( our_rep )
word_mat_dnn = scale( word_mat_dnn )

pdist_dnn = cosine_distances( word_mat_dnn ) #pairwise_distances( word_mat_dnn )


plt.hist( pdist_orig.reshape(-1) - pdist_dnn.reshape(-1))


def compareWordDistance( w1, w2 ):
	"""

	"""	
	ii = wordVector_dict.keys().index(w1)
	ii2 = wordVector_dict.keys().index(w2)

	print 'original distance:' + str(np.sqrt((( word_mat[ii,:] - word_mat[ii2,:])**2).sum()))
	print 'new distance: ' +str(np.sqrt((( word_mat_dnn[ii,:] - word_mat_dnn[ii2,:])**2).sum()))


compareWordDistance('face','emotion')
compareWordDistance('face','amygdala')
compareWordDistance('amygdala','emotion')

compareWordDistance('hippocampus', 'spatial')
compareWordDistance('hippocampus', 'memory')

compareWordDistance('language','broca')
compareWordDistance('tapping','motor')


def seeChange1Word( w1 ):
	"""
	find the words whose distance wrt w1 have changed the most
	"""
	ii = wordVector_dict.keys().index(w1)

	print 'most similar original:\n\t' + '\n\t'.join( np.array(wordVector_dict.keys())[pdist_orig[:,ii].argsort()[1:11]] )

	print 'most similar ours:\n\t' + '\n\t'.join( np.array(wordVector_dict.keys())[pdist_dnn[:,ii].argsort()[1:11]] )



# we can do a scatter plot of non-parametric correlations between word distance (using each word embedding)
# and the predicted brain images - so see which is more correlated! 

word_brain_maps = GetBrainMap( net, word_rep_dnn )
word_brain_maps = scale( word_brain_maps )
ii = np.where( word_brain_maps.mean(axis=1) > np.percentile( word_brain_maps.mean(axis=1), 80))[0] # 50


pdist_brain_maps = cosine_distances( word_brain_maps[ii,:] )
pdist_dnn = cosine_distances( word_mat_dnn[ii,:] )
pdist_orig = cosine_distances( word_mat[ii,:] ) 

# now do the scatter plot:
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.scatter( pdist_dnn[ np.triu_indices(n=len(ii), k=1) ], pdist_brain_maps[ np.triu_indices(n=len(ii), k=1) ])
ax2.scatter( pdist_orig[ np.triu_indices(n=len(ii), k=1) ], pdist_brain_maps[ np.triu_indices(n=len(ii), k=1) ])
ax3.scatter( pdist_dnn[ np.triu_indices(n=len(ii), k=1) ], pdist_orig[ np.triu_indices(n=len(ii), k=1) ])

ax3.plot( np.linspace(0,1.5), np.linspace(0, 1.5), color='red', linewidth=2, linestyle='--')


from scipy.stats import spearmanr
spearmanr( pdist_orig[ np.triu_indices(n=len(ii), k=1) ], pdist_brain_maps[ np.triu_indices(n=len(ii), k=1) ])
# there is some improvement, but not huge..
spearmanr( pdist_dnn[ np.triu_indices(n=len(ii), k=1) ], pdist_brain_maps[ np.triu_indices(n=len(ii), k=1) ])



















# load in data and compare activations across L1 and L2 trained networks!
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/final')
myData = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_testing.p' )
test_loader = DataLoader( myData, batch_size = 128 )

myData_train = neurosynthData( 'MatrixFormated_kernsize_10_pubmedVectors_training.p' )
train_loader = DataLoader( myData, batch_size = 128 )


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




#### apply Aapo idea of learning a new vector representation for words based on the final layer!
#
#

# load in data
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/final')
dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors_training.p', 'rb'))

# put word vectors into a matrix
word_mat = np.zeros(( len(wordVector_dict.keys()), 200))
for k in range(len(wordVector_dict.keys())):
	word_mat[k,:] = wordVector_dict[ wordVector_dict.keys()[k] ]


from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances



# project all word vectors into the final representation
word_rep_dnn = Variable( torch.from_numpy( word_mat ).float())

def GetFinalRepresentation(net, vectors):
	"""
	Get the final hidden representation
	"""

	n = vectors.size()[0]
	net_steps = net.children().next()
	out = vectors.view(-1, 200, 1, 1)
	for i in range(len(net_steps)-2):
		out = net_steps[i](out)

	out = out.view(n, -1).data.numpy()

	return out

our_rep = GetFinalRepresentation( net, word_rep_dnn ) # actually I need to do this on words!
our_rep = np.array(our_rep)

# project into 200 dims
pca_mod = PCA(n_components=200)
pca_mod.fit( our_rep )
word_mat_dnn = pca_mod.transform( our_rep )
word_mat_dnn = scale( word_mat_dnn )
word_mat = scale(word_mat)

# run PCA on this representation:
pdist_dnn = pairwise_distances( word_mat_dnn )
pdist_orig = pairwise_distances( word_mat )

plt.hist( pdist_orig.reshape(-1) - pdist_dnn.reshape(-1))


def compareWordDistance( w1, w2 ):
	"""

	"""	
	ii = wordVector_dict.keys().index(w1)
	ii2 = wordVector_dict.keys().index(w2)

	print 'original distance:' + str(np.sqrt((( word_mat[ii,:] - word_mat[ii2,:])**2).sum()))
	print 'new distance: ' +str(np.sqrt((( word_mat_dnn[ii,:] - word_mat_dnn[ii2,:])**2).sum()))


compareWordDistance('face','emotion')
compareWordDistance('face','amygdala')
compareWordDistance('amygdala','emotion')

compareWordDistance('hippocampus', 'spatial')
compareWordDistance('hippocampus', 'memory')
compareWordDistance('parahippocampal', 'memory')

compareWordDistance('language','broca')

compareWordDistance('tapping','motor')


pdist_delta = (pdist_orig - pdist_dnn) / (pdist_orig+0.0001)

words_closer =  {x:[] for x in wordVector_dict.keys()}
words_further = {x:[] for x in wordVector_dict.keys()}
threshold = .2
for i in range(pdist_delta.shape[0]):
	for j in range(i+1, pdist_delta.shape[0]):
		if pdist_delta[i,j] > threshold:
			words_closer[ wordVector_dict.keys()[i] ].append( wordVector_dict.keys()[j] )
			words_closer[ wordVector_dict.keys()[j] ].append( wordVector_dict.keys()[i] )
		if pdist_delta[i,j] < threshold*-1:
			words_further[ wordVector_dict.keys()[i] ].append( wordVector_dict.keys()[j] )
			words_further[ wordVector_dict.keys()[j] ].append( wordVector_dict.keys()[i] )

