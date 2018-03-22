### Exploratory data analysis
#
#
#

import numpy as np 
import os 
import cPickle as pickle 
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import scale
import pylab as plt; plt.ion()

# load in data:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
dat = pickle.load(open('MatrixFormated_kernsize_10_pubmedVectors.p', 'rb'))

# study word vectors - start with PCA!
wordVec = scale( dat['wordVectors'], with_std=False )
pca_vec = PCA( n_components=3).fit_transform( wordVec )
plt.scatter( pca_vec[:,0], pca_vec[:,1])

# some clustering:
ncluster = 50
sk_clus = SpectralClustering( n_clusters=ncluster ).fit( wordVec )

clusVec = [ np.array( dat['imageVectors'] )[ np.where( sk_clus.labels_==x )[0] ].mean(axis=0).reshape((20,20)) for x in range(ncluster) ]

for x in range(ncluster):
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
	plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1)

	my_arr = clusVec[x]
	my_arr_mask = np.ma.masked_where(my_arr < 0.05, my_arr)

	cmap = plt.cm.YlOrRd
	cmap.set_bad(color='white')
	limits = 100 # 90 
	plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())
	os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Figures/tmp')
	plt.savefig('wordvec_cluster_activity_clus_'+str(x+1)+'.png')
	plt.close()

















