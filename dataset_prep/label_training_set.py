
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
##import pycroscopy as px
##import pyUSID as usid
import skimage
from skimage import io
from skimage import filters
#from skimage.morphology import disk
#from skimage.feature import blob_log
from skimage import measure
from skimage.filters import threshold_mean, threshold_isodata, threshold_minimum, threshold_otsu, threshold_li
#from skimage import exposure
import os
from pycroscopy.io.translators.df_utils import dm_utils
from pycroscopy.io.translators.df_utils import parse_dm3
from scipy import fftpack
#from segmentation import *
#from scipy.stats import norm
import scipy
import pandas as pd
import time
import os
from skimage.morphology import reconstruction, opening, disk
import argparse
import warnings
import glob
warnings.filterwarnings('ignore')


# In[100]:
### Set up for use with GNU Parallel (in terminal)
parser = argparse.ArgumentParser()
parser.add_argument('image_file',type=str)
args = parser.parse_args()

def normalize_image(image):
    return (image - image.min())/image.max()
#

# In[101]:
os.chdir('/home/jay/ExtraDrive/1k_dataset/dm_files')
im_dir = '/home/jay/ExtraDrive/1k_dataset/training_set/dataset/images_full_0404'
label_dir = '/home/jay/ExtraDrive/1k_dataset/training_set/dataset/labels_full_0404'

def process_image(image_file):
    image,p = dm_utils.read_dm4(image_file)  ## Read image
    image = skimage.color.rgb2gray(image)
    image = normalize_image(image)  ## Normalize on range [0,1]
    im_f = normalize_image(filters.gaussian(image,sigma=2))  ## Apply gaussian filter
    
    seed = im_f.copy() ## First reconstruction step
    seed[1:-1,1:-1] = im_f.min()
    mask = im_f
    r = reconstruction(seed,mask,method='dilation')
#    im_s = filters.sobel(r)
    
    
    seed = r.copy()  ##  Second reconstruction;  this step finds the background
    seed[1:-1,1:-1] = r.max()
    mask = r
    r1 = reconstruction(seed,mask,method='erosion')
    
    back_subtract = normalize_image(np.abs(r - r1) + filters.sobel(np.abs(r -r1))) ##  subtract background (r1) from image (r);  apply sobel filter to get better contour at particle edges
    
    o = normalize_image(opening(back_subtract,disk(5)))  ## opening removes small white spots (the size of disk controls the size of the smallest component of the particle).  You can see the impact by looking at label images; some look like particles are made up of small circles.

    plt.imsave(os.path.join(im_dir,image_file[:-3]+'jpg'),image,cmap='gray')
    plt.imsave(os.path.join(label_dir,image_file[:-3]+'jpg'),o>filters.threshold_otsu(o),cmap='gray')
    
    
    
# %%
    
##### Not used here, but another example function for processing raw images

def process_TEM_image(image_file):
    
    image,p = dm_utils.read_dm4(image_file)
    image = skimage.color.rgb2gray(image)
    scale = p['Root_ImageList_SubDir_000_ImageData_Calibrations_Dimension_SubDir_000_Scale']
    
    name = os.path.splitext(os.path.basename(image_file))[0]
    name_entries = name.split('_')
    capture = name_entries[0]
    hour = name_entries[2]
    minute = name_entries[4]
    second = name_entries[6]
    frame = name_entries[8]
    #temp = name[0]
    #hour = name[-7]
    #minute = name[-5]
    #second = name[-3]
    #frame = name[-1]
    
    im = image
    im_cut = im.copy()#[448:1472,384:1408]
    im = normalize_image(im)
    im_f = filters.gaussian(im,sigma = 1.5)
    #e = skimage.exposure.equalize_hist(im_f)
    #e_f = filters.gaussian(e,sigma=10)

    seed = im_f.copy()
    seed[1:-1,1:-1] = im_f.min()
    mask = im_f
    r = reconstruction(seed,mask,method='dilation')
    
    seed = r.copy()
    seed[1:-1,1:-1] = r.max()
    mask = r
    r1 = reconstruction(seed,mask,method='erosion')
    
    #new_image = normalize_image(np.abs(r - r1))
    new_image = np.abs(r - r1)
    new_image = normalize_image(new_image)
    inv = 1 - new_image

    inv_t = new_image > threshold_otsu(new_image)
    inv_t = inv_t#[448:1472,384:1408]
    return im,inv_t
#    label = measure.label(inv_t)

    plt.imsave(os.path.join(im_dir,image_file[:-3]+'jpg'),im_cut,cmap='gray')
    plt.imsave(os.path.join(label_dir,image_file[:-3]+'jpg'),inv_t,cmap='gray')


# In[96]:

process_image(args.image_file)
