### Build joint dataset
#
#
#
#
# the dataset is as follows:
#	- store as a dict of dicts. Each key a pubmed id 
#	- each entry contains vector representation (averaged across all words in the corpus) and downsampled brain activation
#	- also put into matrix format (training data or word vectors and image response)
#
#
#

import numpy as np 
import pandas as pd 
import cPickle as pickle 
import os 
#from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import gensim # for word vector embeddings!

os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/utils')
from plotBrains import *  # contains functions to downsample images!

# define some parameters:
kernelSize = 10 # width of Gaussian kernel (used for images)
downsampleSize = 10 # downsampling ratio
tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))
#poldrack_stopwords = list( pd.read_table( 'stopwords.txt').iloc[:,0])

# load word embeddings - note that we use pubmed embeddings based on
# the following paper: http://bio.nlplab.org/
model = model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ricardo/Downloads/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

def cleanAbstract( abstract ):
	"""
	we tokenize, remove stop words and return abstract as a list
	"""

	tokens = tokenizer.tokenize( abstract )
	tokens = [ x.lower() for x in tokens ]
	tokens = [ x for x in tokens if x not in stopWords  ]
	# remove poldrack stopwords as well
	#tokens = [ x for x in tokens if x  in poldrack_stopwords ]

	return tokens 

def getMeanVectorRepresentation( tokens ):
	"""
	compute the mean word embedding vector for all words in the abstract 
	"""
	abs_vec = np.zeros((200,))
	counter = 0
	for x in tokens:
		try: 
			abs_vec += model.get_vector(x)
			counter += 1
		except KeyError:
			pass

	return abs_vec/counter



# load in data
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
res = pickle.load(open('Abstract_MNI_raw.p', 'rb'))

dataVec = {} # will populate this dictionary as we go

for pid in res.keys():
	if len( res[pid]['abstract'] ) == 0:
		pass
	else:
		mni_coords = np.array(res[pid]['MNI'])
		dataVec[pid] = {'image': downsample2d(Get_2d_smoothed_activation( mni_coords, kernelSize),downsampleSize).T[::-1,:], 
						'wordvec': getMeanVectorRepresentation( cleanAbstract( res[pid]['abstract'] ) )}


os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
pickle.dump( dataVec, open('Dict_Vectorized_Downsampled_kernsize_'+str(kernelSize)+'_pubmedVectors.p', 'wb'))

# also put into matrix format
vecMat = np.zeros(( len(dataVec.keys()), 200 ))
imageMat = np.zeros(( len(dataVec.keys()), 400 ))

for x in range(len(dataVec.keys())):
	vecMat[x,:] = dataVec[ dataVec.keys()[x] ]['wordvec']
	imageMat[x,:] = dataVec[ dataVec.keys()[x] ]['image'].reshape(-1)


pickle.dump( {'wordVectors': vecMat, 'imageVectors': imageMat, 'pid':dataVec.keys()}, open('MatrixFormated_kernsize_'+str(kernelSize)+'_pubmedVectors.p', 'wb'))


