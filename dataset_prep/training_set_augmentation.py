# %%

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import*
import os

# %% 

##im_dir = (r'C:\Users\Jay Horwath\Desktop\NN_TEST\training_set\images1')
##lab_dir = (r'C:\Users\Jay Horwath\Desktop\NN_TEST\training_set\labels1')
##aug_im = (r'C:\Users\Jay Horwath\Desktop\NN_TEST\training_set\aug_im')
##aug_lab = (r'C:\Users\Jay Horwath\Desktop\NN_TEST\training_set\aug_lab')

im_dir = (r'/home/jay/Desktop/training_set/images1')
lab_dir = (r'/home/jay/Desktop/training_set/labels1')
aug_im = (r'/home/jay/Desktop/training_set/augim1_0315')
aug_lab = (r'/home/jay/Desktop/training_set/auglab1_0315')

# %%

for i in range(len(os.listdir(im_dir))):
#for i in range(2):
    print('{}/{}'.format(i+1,len(os.listdir(im_dir))))
    im = skimage.color.rgb2gray(plt.imread(os.path.join(im_dir,os.listdir(im_dir)[i])))
    lab = skimage.color.rgb2gray(plt.imread(os.path.join(lab_dir,os.listdir(lab_dir)[i])))

    n_x = int(np.random.rand()*384) ## 448 = 512/8, so that shift can be by almost entire image dimension
    n_y = int(np.random.rand()*384)
    n_x1 = int(np.random.rand()*384)
    n_y1 = int(np.random.rand()*384)
    r1 = np.random.rand()
    r2 =np.random.rand()
    r3 =np.random.rand()
    r4 =np.random.rand()
    r5 =np.random.rand()
    r6 = np.random.rand()

    #### Augment Image
##    im = im[:1792,:]
    os.chdir(aug_im)
    im_c = im[n_x:n_x+128,n_y:n_y+128]
    plt.imsave('{}_zoom.png'.format(i),resize(im_c,(512,512)),cmap='gray')
    im_c = im[n_x1:n_x1+128,n_y1:n_y1+128]
    plt.imsave('{}_zoom1.png'.format(i),resize(im_c,(512,512)),cmap='gray')
##    im = im[:1792,:]
    #im = resize(im,(512,512))
    plt.imsave('{}_base.png'.format(i),im,cmap='gray')

##    flip1 = im[:,::-1]
##    flip2 = im[::-1,::-1]

    t1 = AffineTransform(translation = (int(r1*im.shape[0]),int(r2*im.shape[1])))
    t2 = AffineTransform(translation = (int(r3*im.shape[0]),int(r4*im.shape[1])))
    t3 = AffineTransform(translation = (int(r5*im.shape[0]),int(r6*im.shape[1])))
    
    shift1 = warp(im,t1,mode='wrap')
    shift2 = warp(im,t2,mode='wrap')
    shift3 = warp(im,t3,mode='wrap')

    plt.imsave('{}_shift1.png'.format(i),shift1,cmap='gray')
    plt.imsave('{}_shift2.png'.format(i),shift2,cmap='gray')
    plt.imsave('{}_shift3.png'.format(i),shift3,cmap='gray')


    ###Augment Labels

    os.chdir(aug_lab)
    lab_c = lab[n_x:n_x+128,n_y:n_y+128]
    plt.imsave('{}_zoom.png'.format(i),resize(lab_c,(512,512)),cmap='gray')
    lab_c = lab[n_x1:n_x1+128,n_y1:n_y1+128]
    plt.imsave('{}_zoom1.png'.format(i),resize(lab_c,(512,512)),cmap='gray')
##    im = im[:1792,:]
    #im = resize(lab,(512,512))
    plt.imsave('{}_base.png'.format(i),lab,cmap='gray')

##    flip1 = im[:,::-1]
##    flip2 = im[::-1,::-1]

    t1 = AffineTransform(translation = (int(r1*im.shape[0]),int(r2*im.shape[1])))
    t2 = AffineTransform(translation = (int(r3*im.shape[0]),int(r4*im.shape[1])))
    t3 = AffineTransform(translation = (int(r5*im.shape[0]),int(r6*im.shape[1])))
    
    shift1 = warp(lab,t1,mode='wrap')
    shift2 = warp(lab,t2,mode='wrap')
    shift3 = warp(lab,t3,mode='wrap')

    plt.imsave('{}_shift1.png'.format(i),shift1,cmap='gray')
    plt.imsave('{}_shift2.png'.format(i),shift2,cmap='gray')
    plt.imsave('{}_shift3.png'.format(i),shift3,cmap='gray')
