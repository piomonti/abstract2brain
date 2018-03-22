## Plotting filtered activations
#
#
# This code has been shamelessly taken from: 
# https://github.com/poldrack/LatentStructure/blob/master/src/utils/foci_to_image.py
#
# read in foci and create a nifti image
# 
# for now we read from text file
# TBD: read directly from mysql db
#
#
#
# NOTES:
#   - to run this file do the following:
#       1) run: 
#               foci_to_image( MNI_coords, filename )
#          this will save a .nii.gz file with the Gaussian kernel convolution
#       2) then run:
#               from nilearn import plotting
#               plotting.plot_glass_brain( filename ) 
#          this will make a plot of the activations!
#   
#
# OTHER NOTES:
#   - project onto x/y plane (i.e. ignore z plane)
#   - use: from scipy.ndimage.filters import gaussian_filter
#         i.e.,: gaussian_filter( arr, sigma=10)     
#


import numpy as N
import numpy # cba to change
import numpy as np
import nibabel as nb
import subprocess as sub
import os
import scipy.signal as sig
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter

def convert_MNI_to_voxel_coords(MNIcoord, resolution):
    # first check c
    if len(MNIcoord)!=3:
        raise IndexError

    # inv qform obtained from MNI152lin_T1_2mm_brain.nii.gz
    if resolution==2:
      invQForm=N.array([[ -0.5,  -0. ,  -0. ,  45. ],
       [  0. ,   0.5,   0. ,  63. ],
       [  0. ,   0. ,   0.5,  36. ],
       [  0. ,   0. ,   0. ,   1. ]])
    elif resolution==3:
      invQForm=N.array([[ -0.3333,0,0,30.0000],
         [0,0.3333,0,42.0000],
         [0,0,0.3333,24.0000],
         [0,0,0,1.0000]])
    else:
	    print '%dmm resolution not supported - exiting'%resolution
	    return


    # needs to be a homogenous coordinate array so add an extra 1 
    # in the 4th position
    coord_array=N.ones((1,4))
    coord_array[0][0:3]=MNIcoord

    trans_coord=N.dot(invQForm,coord_array.T)[:][0:3].T[0]

    return trans_coord

def foci_to_image(all_MNI_coords,outfileprefix,kernel='sphere',radius=10):
    try:
        FSLDIR=os.environ['FSLDIR']
    except:
        print 'it appears that FSL has not been configured'
        print 'you should set FSLDIR and then source $FSLDIR/etc/fslconf/fsl.{sh,csh}'
        return

# TBD: check foci format
# should be N x 3 array
    tmpdir='/tmp/'
    # use 3mm template to reduce data size
    mni_template='/Users/ricardo/Documents/Projects/neurosynth_dnn/Data/MNI152_T1_3mm_brain.nii.gz'

    data=N.zeros((60,72,60))

    #####################
    # read coordinates and add to dataset

    for cnum in range(len(all_MNI_coords)):
        MNI_coords=all_MNI_coords[cnum]
        voxel_coords=convert_MNI_to_voxel_coords(MNI_coords,3).astype('int')
        #if validate_voxel_coords(voxel_coords,3)==0:
            #print 'coord not valid:', voxel_coords
        #    continue
        #else:
        data[voxel_coords[0]][voxel_coords[1]][voxel_coords[2]]=1


    # KLUDGE: I've not been able to find any python code to do 3d convolution
    # so I am using an external call to fslmaths

    ###########################
    # save the unconvolved image
    mni_image=nb.load(mni_template)
    new_image=nb.Nifti1Image(data,mni_image.get_affine(),header=mni_image.get_header())
    tmpfname=tmpdir+'tmp_%.8f.nii.gz'%N.random.rand()
    new_image.to_filename(tmpfname)

    #####################
    # convolve points with sphere
    filt_tmpfname=tmpfname.replace('.nii.gz','_filt.nii.gz')
    if kernel=='gauss':
        cmd='fslmaths %s -kernel gauss %f -fmean %s'%(tmpfname,radius,filt_tmpfname)
    else:
        cmd='fslmaths %s -kernel sphere %f -fmean -bin %s'%(tmpfname,radius,outfileprefix)
    #print cmd
    p = sub.Popen(cmd,stdout=sub.PIPE,stderr=sub.PIPE,shell=True)
    output, errors = p.communicate()

    # clean up temp file
    if os.path.isfile(tmpfname):
        os.remove(tmpfname)
    # load the filtered image and ensure max = 1
    if kernel=='gauss':
        filt_image=nb.load(filt_tmpfname)
        filtdata=filt_image.get_data()
        filtdata=filtdata/N.max(filtdata)
        final_image=nb.Nifti1Image(filtdata,mni_image.get_affine(),header=mni_image.get_header())
        final_image.to_filename(outfileprefix+'.nii.gz')
        # clean up temp file
        if os.path.isfile(filt_tmpfname):
            os.remove(filt_tmpfname)
        


def Get_2d_smoothed_activation( MNI_coords, kernel_width=10 ):
    """
    project onto z=0 and get smoothed activation!
    """
    MNI_coords = MNI_coords[:, :2].astype('int') + 100

    arr = np.zeros((200,200))
    arr[ MNI_coords[:,0], MNI_coords[:,1]] = 1

    return gaussian_filter( arr, kernel_width ) 


def downsample2d(inputArray, kernelSize):
    """This function downsamples a 2d numpy array by convolving with a flat
    kernel and then sub-sampling the resulting array.
    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.
    :param: inputArray: 2d numpy array
    :param: kernelSize: integer

    this has been shamelessly taken from: https://gist.github.com/andrewgiessel/2955714
    """
    average_kernel = np.ones((kernelSize,kernelSize))

    blurred_array = sig.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array



def upsampleImage( arr, kernelSize ):
    """
    from a low dimensional image we sample up!s
    """
    return scipy.ndimage.zoom( arr, kernelSize )


