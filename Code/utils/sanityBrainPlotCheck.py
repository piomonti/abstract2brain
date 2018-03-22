### SANITY CHECK BRAIN PLOTTING
#
#
#

import numpy as np 
import pandas as pd 
import cPickle as pickle 
from scipy.ndimage.filters import gaussian_filter
import os 
from nilearn import plotting
import pylab as plt; plt.ion()

# load in code:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Code/utils')
from plotBrains import * 

# load in the data:
os.chdir('/Users/ricardo/Documents/Projects/neurosynth_dnn/Data')
res = pickle.load(open('Abstract_MNI_raw.p', 'rb'))

mni_coords = res[ res.keys()[500] ]['MNI']


# run nilearn plotting:
foci_to_image( np.array(mni_coords), 'trial_name' )
plotting.plot_glass_brain( 'trial_name' + '.nii.gz' ); plotting.show()

plotting.plot_glass_brain( 'trial_name.nii.gz', display_mode='z', threshold=1)
ksize = 10
my_arr = downsample2d(Get_2d_smoothed_activation(np.array(mni_coords), ksize),ksize).T[::-1,:]
my_arr_mask = np.ma.masked_where(my_arr < 0.05, my_arr)

cmap = plt.cm.YlOrRd
cmap.set_bad(color='white')
limits = 100 # 90 
plt.imshow( my_arr_mask , extent=(-1* limits, limits, -1*limits, limits), cmap=cmap, vmin=0)#, vmax=my_arr.max())














