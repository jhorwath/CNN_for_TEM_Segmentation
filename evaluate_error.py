#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:03:00 2019

@author: jay
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import skimage
from skimage import filters, measure
from skimage.measure import label,regionprops
import pandas as pd

# %%

model_results = np.load('all_model_results_softmax.npy')
test_im = skimage.color.rgb2gray(plt.imread('test_image.tif'))
test_lab = np.load('test_label.npy')

man_j,man_k = pd.read_csv('test_image_measure.csv'),pd.read_csv('Test_image_results.csv')
psd_j,psd_k = man_j.Length,man_k.Length

print(model_results.shape,test_im.shape,test_lab.shape)


# =============================================================================
# Compare Histograms for particle measurements by different models
# =============================================================================
# %%

im = model_results[0,4,1]
im_otsu = im > filters.threshold_otsu(im)
im_70 = im > 0.7

l = [test_lab,model_results[0,2,1] > filters.threshold_otsu(model_results[0,2,1]),im_otsu,im_70]
p = {}

f,a = plt.subplots(figsize=(20,20))
for i in range(len(l)):
    k = l[i]
    lab = label(k)
    psd = []
    for region in regionprops(lab):
        psd.append(region.major_axis_length * 0.18)
    
    p['{}'.format(i)] = psd
    
    a.hist(psd,label='{}'.format(i),alpha = (1 - 0.15*i))
    
a.legend()

# %%

plt.subplots(figsize=(20,20))
plt.hist(p['0'],label='training',bins=10)
plt.hist(psd_j,alpha=0.8,label='j',bins=10)
plt.hist(psd_k,alpha=0.7,label='k',bins=10)
plt.legend()

# %%

l1 = [test_lab,model_results[0,2,1] > filters.threshold_otsu(model_results[0,2,1]),im_otsu,im_70,psd_j,psd_k]
l_names = ['test_set','otsu_2ConvBatch','otsu_Batch','t70_Batch','j','k']

p = {}

f,a = plt.subplots(figsize=(20,20))

for i in range(len(l1)):
    k = l1[i]
    
    if k.shape != (1024,1024):
        
        psd = k
        p['{}'.format(l_names[i])] = psd
        counts,bin_edges = np.histogram(psd,normed=True)
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        plt.plot(bin_centers,counts,label='{}'.format(l_names[i]))
        continue
    
    lab = label(k)
    psd = []
    for region in regionprops(lab):
        psd.append(region.major_axis_length * 0.18)
        
        
    p['{}'.format(l_names[i])] = psd
    counts,bin_edges = np.histogram(psd,normed=True)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    plt.plot(bin_centers,counts,label='{}'.format(l_names[i]))
    
a.legend()

# %%
# =============================================================================
# Estimate error between 50 training set images and model results
# =============================================================================

test_lab = np.load('/home/jay/data/no_blur/test_labels.npy')[:50]
test_im = np.load('/home/jay/data/no_blur/test_images.npy')[:50]

model_results = np.load('50_results_2ConvNormAndNorm.npy')
model_soft = np.load('50_results_2ConvNormAndNorm_soft.npy')

# %%

f,a = plt.subplots(ncols=2,figsize=(20,10))
for i in range(2):
    a[i].imshow(test_im[10])
a[0].imshow(model_soft[0,10,1],plt.cm.jet,alpha=0.3)
a[1].imshow(model_soft[1,10,1],plt.cm.jet,alpha=0.3)

# %%
t_2convnorm = filters.threshold_otsu(model_soft[0,10,1])
t_norm = filters.threshold_otsu(model_soft[1,10,1])
f,a = plt.subplots(ncols=2,figsize=(40,20))
for i in range(2):
    a[i].imshow(test_im[10])
a[0].imshow(model_soft[0,10,1] > t_2convnorm,plt.cm.jet,alpha=0.3)
a[0].set_title('{}'.format(t_2convnorm),fontsize=24)
a[1].imshow(model_soft[1,10,1] > t_norm,plt.cm.jet,alpha=0.3)
a[1].set_title('{}'.format(t_norm),fontsize=24)


# %%
thresh = np.zeros((2,50,1024,1024))
t_vals = {}
for i in range(len(model_soft)):
    t = []
    for j in range(len(model_soft[i])):
        im = model_soft[i,j,1]
        otsu_t = filters.threshold_otsu(im)
        print(otsu_t)
        t.append(otsu_t)
        thresh[i,j] = im > otsu_t
    t_vals['{}'.format(i)] = t
    
# %%

f,a = plt.subplots(figsize=(20,20))
for i in t_vals.keys():
    t = t_vals[i]
    a.hist(t,label=i)
a.legend()

# %%
error = {}
for i in range(len(thresh)):
    e = []
    for j in range(len(thresh[i])):
        m = thresh[i,j].reshape(1024*1024,1)
        t = test_lab[j].reshape(1024*1024,1)
        e.append(np.sum(np.abs(m-t)))
        error['{}'.format(i)] = e
    print('{}: mean error = {}'.format(i,np.mean(e)))
    
# %%
error_vs_thresh = {}
thresh_images = {}
for i in range(10):
    thresh_val = i/10
    t = np.zeros((2,50,1024,1024))
    for j in range(len(model_soft)):
        e = []
        for k in range(len(model_soft[j])):
            im = model_soft[j,k,1]
            ref = test_lab[k].reshape(1024**2,1)
            im_t = im > thresh_val
            e.append(np.sum(np.abs(im_t.reshape(1024**2,1)-ref)))
            t[j,k] = im_t
        error_vs_thresh['{}_{}'.format(i,j)] = e
    thresh_images['{}'.format(i)] = t
    
# %%
    
mean_error = np.zeros((10,2))
n = 0
labs = ['NoBlur_2ConvNorm','NoBlur_NormOnly']
for i in error_vs_thresh.keys():
    if '_0' in i:
        k = 0
        mean_error[n,k] = np.mean(error_vs_thresh[i])/1024**2
    else:
        k = 1
        mean_error[n,k] = np.mean(error_vs_thresh[i])/1024**2
        n += 1
x_vals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
f,a = plt.subplots(figsize=(20,20))
for i in range(2):
    vals = mean_error[:,i]
    a.plot(x_vals,vals,label=labs[i])
a.legend(fontsize=24)
a.set_xlabel('Threshold Value',fontsize=24)
a.set_ylabel('Mean Validation Error',fontsize=24)
f.savefig('threshold_error_comparison.png')


# %%
# =============================================================================
# Line Scan Across selected particles
# =============================================================================
images = [skimage.color.rgb2gray(plt.imread('test_image.tif')),np.load('test_2ConvNorm.npy'),np.load('test_Norm.npy')]
test_image = skimage.color.rgb2gray(plt.imread('test_image.tif'))
colors=['red','blue','black','magenta']

lines = np.array([[348,439,100],
                 [476,487,100],
                 [272,780,100],
                 [589,108,72]])

f,a = plt.subplots(ncols=2,figsize=(30,20))
a[0].imshow(test_image,plt.cm.gray)
a[1].imshow(test_image,plt.cm.gray)
im = a[0].imshow(images[0],plt.cm.jet,alpha=0.3,vmin=0,vmax=1)
im1 = a[1].imshow(images[1],plt.cm.jet,alpha=0.3,vmin=0,vmax=1)
a[0].set_title('2ConvNorm',fontsize=24)
a[1].set_title('NormOnly',fontsize=24)
f.colorbar(im,ax=a[0],fraction=0.046, pad=0.04)
f.colorbar(im1,ax=a[1],fraction=0.046,pad=0.04)
f.savefig('ModelEval_Part1.png')

fig = plt.figure(figsize=(30,30))
grid = plt.GridSpec(4,5,hspace=0.2,wspace=0.2)
main_ax = fig.add_subplot(grid[:,:4])
main_ax.imshow(test_image,plt.cm.gray)
for i in range(len(lines)):
    main_ax.hlines(lines[i,1],lines[i,0],lines[i,0]+lines[i,2],label='{}'.format(i),color=colors[i],linewidth=4)
for i in range(len(images)):
    if i == 0:
        l = 'solid'
        lab = 'Raw'
    elif i == 1:
        l = 'dashed'
        lab = '2ConvNorm'
    elif i == 2:
        l = 'dotted'
        lab = 'Norm'
    im = images[i]
    for j in range(len(lines)):
        a = fig.add_subplot(grid[j,4])
        a.plot(im[lines[j,1],lines[j,0]:lines[j,0]+lines[j,2]],label=lab,color=colors[j],linestyle=l)
        a.legend()
fig.savefig('ModelEval_Part2.png')