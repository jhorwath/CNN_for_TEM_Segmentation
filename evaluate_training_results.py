#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:42:48 2019

@author: jay
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder
print(torch.__version__)
import time
from skimage import filters, measure

raw_images = np.load('/home/jay/data/no_blur/test_images.npy')[:50]
label = np.load('/home/jay/data/no_blur/test_labels.npy')[:50]                                                                                                                                                                                                                                                                                                       
# %%

#f,a = plt.subplots(figsize=(20,20))
#a.imshow(raw_images[0],plt.cm.gray)
#
## %%
#
#test_raw = raw_images[0]
#test_blur1 = skimage.filters.gaussian(test_raw,sigma=1)
#test_blur2 = skimage.filters.gaussian(test_raw,sigma=2)
#
#f,a = plt.subplots(ncols=3,figsize=(40,20))
#a[0].imshow(test_raw,plt.cm.gray)
#a[1].imshow(test_blur1,plt.cm.gray)
#a[2].imshow(test_blur2,plt.cm.gray)


# %%


no_blur = ['no_blur/2conv_lr0001_NoDecay','no_blur/2conv_lr001_norm_NoDecay','no_blur/2conv_lr0001_norm_NoDecay','no_blur/norm_lr001_NoDecay','no_blur/norm_lr0001_NoDecay']
blur1 = ['train_sig1/2conv_lr0001_NoDecay','train_sig1/2conv_lr001_norm_NoDecay','train_sig1/2conv_lr0001_norm_NoDecay','train_sig1/norm_lr001_NoDecay','train_sig1/norm_lr0001_NoDecay']
blur2 = ['train_sig2/2conv_lr0001_NoDecay','train_sig2/2conv_lr001_norm_NoDecay','train_sig2/2conv_lr0001_norm_NoDecay','train_sig2/norm_lr001_NoDecay','train_sig2/norm_lr0001_NoDecay']


# %%

class twoconv_only(nn.Module):
    def __init__(self):
        super(twoconv_only,self).__init__()
        self.conv0 = nn.Conv2d(1,8,3,dilation=1,padding=1)
        self.conv1 = nn.Conv2d(8,16,3,dilation=1,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,dilation=1,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,dilation=1,padding=1)
        self.conv4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=(1,1))
        self.conv4_1 = nn.Conv2d(96,96,3,padding=1)
        self.conv5 = nn.ConvTranspose2d(96,32,3,stride=2,padding=(1,1))
        self.conv5_1 = nn.Conv2d(64,64,3,padding=1)
        self.conv6 = nn.ConvTranspose2d(64,16,3,stride=2,padding=(1,1))
        self.conv6_1 = nn.Conv2d(32,32,3,padding=1)
        self.conv7 = nn.ConvTranspose2d(32,16,3,stride=2,padding=(1,1))
        self.conv7_1 = nn.Conv2d(24,24,3,padding=1)
        self.conv8 = nn.Conv2d(24,8,3,padding=(1,1))
        self.conv8_1 = nn.Conv2d(8,8,3,padding=1)
        self.conv9 = nn.Conv2d(8,2,1)#,padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        #self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self,x):
        c0 = F.leaky_relu(self.conv0(x))
        c0out = self.pool(c0)
        c1 = F.leaky_relu(self.conv1(c0out))
        c1out = self.pool(c1)
        c2 = F.leaky_relu(self.conv2(c1out))
        c2out = self.pool(c2)
        c3 = F.leaky_relu(self.conv3(c2out))
        c3out = self.pool(c3)
        c4 = F.leaky_relu(self.conv4(c3out,output_size=c3.size()))
        c4cat = torch.cat((c4,c3),dim=1)
        c4cat = F.leaky_relu(self.conv4_1(c4cat))
        c5 = F.leaky_relu(self.conv5(c4cat,output_size=c2.size()))
        c5cat = torch.cat((c5,c2),dim=1)
        c5cat = F.leaky_relu(self.conv5_1(c5cat))
        c6 = F.leaky_relu(self.conv6(c5cat,output_size=c1.size()))
        c6cat = torch.cat((c6,c1),dim=1)
        c6cat = F.leaky_relu(self.conv6_1(c6cat))
        c7 = F.leaky_relu(self.conv7(c6cat,output_size=c0.size()))
        c7cat = torch.cat((c7,c0),dim=1)
        c7cat = F.leaky_relu(self.conv7_1(c7cat))
        c8 = F.leaky_relu(self.conv8(c7cat))
        c8 = F.leaky_relu(self.conv8_1(c8))
        c9 = F.leaky_relu(self.conv9(c8))
#        
        return c9
    
class twoconv_norm(nn.Module):
    def __init__(self):
        super(twoconv_norm,self).__init__()
        self.conv0 = nn.Conv2d(1,8,3,dilation=1,padding=1)
        self.b0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8,16,3,dilation=1,padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,dilation=1,padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,dilation=1,padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=(1,1))
        self.b4 = nn.BatchNorm2d(32)
        self.conv4_1 = nn.Conv2d(96,96,3,padding=1)
        self.b4_1 = nn.BatchNorm2d(96)
        self.conv5 = nn.ConvTranspose2d(96,32,3,stride=2,padding=(1,1))
        self.b5 = nn.BatchNorm2d(32)
        self.conv5_1 = nn.Conv2d(64,64,3,padding=1)
        self.b5_1 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64,16,3,stride=2,padding=(1,1))
        self.b6 = nn.BatchNorm2d(16)
        self.conv6_1 = nn.Conv2d(32,32,3,padding=1)
        self.b6_1 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32,16,3,stride=2,padding=(1,1))
        self.b7 = nn.BatchNorm2d(16)
        self.conv7_1 = nn.Conv2d(24,24,3,padding=1)
        self.b7_1 = nn.BatchNorm2d(24)
        self.conv8 = nn.Conv2d(24,8,3,padding=(1,1))
        self.b8 = nn.BatchNorm2d(8)
        self.conv8_1 = nn.Conv2d(8,8,3,padding=1)
        self.b8_1 = nn.BatchNorm2d(8)
        self.conv9 = nn.Conv2d(8,2,1)#,padding=(1,1))
        self.b9 = nn.BatchNorm2d(2)
        self.pool = nn.MaxPool2d(2,2)
        #self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self,x):
        c0 = self.b0(F.leaky_relu(self.conv0(x)))
        c0out = self.pool(c0)
        c1 = self.b1(F.leaky_relu(self.conv1(c0out)))
        c1out = self.pool(c1)
        c2 = self.b2(F.leaky_relu(self.conv2(c1out)))
        c2out = self.pool(c2)
        c3 = self.b3(F.leaky_relu(self.conv3(c2out)))
        c3out = self.pool(c3)
        c4 = self.b4(F.leaky_relu(self.conv4(c3out,output_size=c3.size())))
        c4cat = torch.cat((c4,c3),dim=1)
        c4cat = self.b4_1(F.leaky_relu(self.conv4_1(c4cat)))
        c5 = self.b5(F.leaky_relu(self.conv5(c4cat,output_size=c2.size())))
        c5cat = torch.cat((c5,c2),dim=1)
        c5cat = self.b5_1(F.leaky_relu(self.conv5_1(c5cat)))
        c6 = self.b6(F.leaky_relu(self.conv6(c5cat,output_size=c1.size())))
        c6cat = torch.cat((c6,c1),dim=1)
        c6cat = self.b6_1(F.leaky_relu(self.conv6_1(c6cat)))
        c7 = self.b7(F.leaky_relu(self.conv7(c6cat,output_size=c0.size())))
        c7cat = torch.cat((c7,c0),dim=1)
        c7cat = self.b7_1(F.leaky_relu(self.conv7_1(c7cat)))
        c8 = self.b8(F.leaky_relu(self.conv8(c7cat)))
        c8 = self.b8_1(F.leaky_relu(self.conv8_1(c8)))
        c9 = self.b9(F.leaky_relu(self.conv9(c8)))
#        
        return c9
    
class norm_only(nn.Module):
    def __init__(self):
        super(norm_only,self).__init__()
        self.conv0 = nn.Conv2d(1,8,3,dilation=1,padding=1)
        self.b0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8,16,3,dilation=1,padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,dilation=1,padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,dilation=1,padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=(1,1))
        self.b4 = nn.BatchNorm2d(32)
        self.conv4_1 = nn.Conv2d(96,96,3,padding=1)
        self.b4_1 = nn.BatchNorm2d(96)
        self.conv5 = nn.ConvTranspose2d(96,32,3,stride=2,padding=(1,1))
        self.b5 = nn.BatchNorm2d(32)
        self.conv5_1 = nn.Conv2d(64,64,3,padding=1)
        self.b5_1 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64,16,3,stride=2,padding=(1,1))
        self.b6 = nn.BatchNorm2d(16)
        self.conv6_1 = nn.Conv2d(32,32,3,padding=1)
        self.b6_1 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32,16,3,stride=2,padding=(1,1))
        self.b7 = nn.BatchNorm2d(16)
        self.conv7_1 = nn.Conv2d(24,24,3,padding=1)
        self.b7_1 = nn.BatchNorm2d(24)
        self.conv8 = nn.Conv2d(24,8,3,padding=(1,1))
        self.b8 = nn.BatchNorm2d(8)
        self.conv8_1 = nn.Conv2d(8,8,3,padding=1)
        self.b8_1 = nn.BatchNorm2d(8)
        self.conv9 = nn.Conv2d(8,2,1)#,padding=(1,1))
        self.b9 = nn.BatchNorm2d(2)
        self.pool = nn.MaxPool2d(2,2)
        #self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self,x):
        c0 = self.b0(F.leaky_relu(self.conv0(x)))
        c0out = self.pool(c0)
        c1 = self.b1(F.leaky_relu(self.conv1(c0out)))
        c1out = self.pool(c1)
        c2 = self.b2(F.leaky_relu(self.conv2(c1out)))
        c2out = self.pool(c2)
        c3 = self.b3(F.leaky_relu(self.conv3(c2out)))
        c3out = self.pool(c3)
        c4 = self.b4(F.leaky_relu(self.conv4(c3out,output_size=c3.size())))
        c4cat = torch.cat((c4,c3),dim=1)
        #c4cat = self.b4_1(F.leaky_relu(self.conv4_1(c4cat)))
        c5 = self.b5(F.leaky_relu(self.conv5(c4cat,output_size=c2.size())))
        c5cat = torch.cat((c5,c2),dim=1)
        #c5cat = self.b5_1(F.leaky_relu(self.conv5_1(c5cat)))
        c6 = self.b6(F.leaky_relu(self.conv6(c5cat,output_size=c1.size())))
        c6cat = torch.cat((c6,c1),dim=1)
        #c6cat = self.b6_1(F.leaky_relu(self.conv6_1(c6cat)))
        c7 = self.b7(F.leaky_relu(self.conv7(c6cat,output_size=c0.size())))
        c7cat = torch.cat((c7,c0),dim=1)
        #c7cat = self.b7_1(F.leaky_relu(self.conv7_1(c7cat)))
        c8 = self.b8(F.leaky_relu(self.conv8(c7cat)))
        #c8 = self.b8_1(F.leaky_relu(self.conv8_1(c8)))
        c9 = self.b9(F.leaky_relu(self.conv9(c8)))
#        
        return c9



# %%

os.chdir('/home/jay/data')

models = [no_blur,blur1,blur2]
#test_images = [test_raw,test_blur1,test_blur2]

#no_blur_results = np.zeros((50,2,1024,1024))
#blur1_results = np.zeros_like(no_blur_results)
#blur2_results = np.zeros_like(no_blur_results)
#total_results = [no_blur_results,blur1_results,blur2_results]
#total_results_softmax = total_results.copy()

total_results = np.zeros((3,5,50,1024,1024))

tensor_im = torch.Tensor(raw_images.reshape(50,1,1024,1024))
dataset = torch.utils.data.TensorDataset(tensor_im)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=50,shuffle=False)

for i in range(len(models)):
    print(i)
    m = models[i]
#    if i > 0:
#        break
#    image = test_images[i].reshape(1,1,test_images[i].shape[0],test_images[i].shape[1])
    for j in range(len(m)):
        print('j: ',m[j])
#        if j > 0:
#            break
#        if m[j] != 'no_blur/2conv_lr0001_norm_NoDecay' and m[j] != 'no_blur/norm_lr0001_NoDecay': 
#            continue
#        print(j)
#        print(i[j])
        os.chdir(m[j])
        
        checkpoint = glob.glob('*25.pt')[0]
        print(checkpoint)
        if '2conv' in checkpoint and 'norm' in checkpoint:
            model = twoconv_norm()
        if '2conv' in checkpoint and 'norm' not in checkpoint:
            model = twoconv_only()
        if 'norm' in checkpoint and '2conv' not in checkpoint:
            model = norm_only()
            
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: ',device)
        model.to(device)
        model = nn.DataParallel(model)
            
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        for k,data in enumerate(dataloader,0):
            inputs = data[0]
            with torch.no_grad():
                p_soft = F.softmax(model(inputs.cuda())).detach().cpu().numpy()
        total_results[i][j] = p_soft[:,1]
        torch.cuda.empty_cache()
            
        
#        inputs = torch.Tensor(image)
#        inputs = torch.Tensor(raw_images.reshape(len(raw_images),1,1024,1024))
#        print(inputs.size())
#        outputs = model(inputs)
#        
#        total_results[i][j] = outputs.detach().cpu().numpy()
#        total_results_softmax[i][j] = F.softmax(outputs).detach().cpu().numpy()
#        
        
        os.chdir('../..')
    print('\n')
    
# %%
    
np.save('/home/jay/data/model_comparison/total_results_all_models.npy',total_results)

# %%

#### error_stats dims: blur, model, thresh, [average, standard deviation],[precision, recall, F1, support]
error_stats = np.zeros((3,5,11,2,4))


for blur in range(total_results.shape[0]):
    print('blur: ',blur)
    for model in range(total_results.shape[1]):
        print('model: ',model)
        for thresh in range(error_stats.shape[2]):
            print('thresh: ',thresh)
            if thresh == 10:
                data = total_results[blur,model] > filters.threshold_otsu(total_results[blur,model])
            else:
                data = total_results[blur,model] > (np.round(thresh/10 + 0.1,2))
            stats = np.zeros((50,4))
            for image in range(stats.shape[0]):
                l = label[image].astype(bool).reshape(1024*1024,1)
                pred = data[image].reshape(1024*1024,1)
                a = precision_recall_fscore_support(l,pred,average='binary')
                stats[image,:] = [a[0],a[1],a[2],a[3]]
            error_stats[blur,model,thresh,0,:] = np.mean(stats,axis=0)
            error_stats[blur,model,thresh,1,:] = np.std(stats,axis=0)
            

        print('\n')
        
        
# %%
#### Plot accuracy and recall for norm only, lr = 0.0001
n = error_stats[1,4]
thresh = np.arange(10)/10 + 0.1
colors = ['red','blue','black']
blurs = ['no blur','blur 1','blur 2']

f,a = plt.subplots(figsize=(20,20))

for i in range(3):
    n = error_stats[i,2]
    a.errorbar(thresh,n[:-1,0,0],yerr=n[:-1,1,0],label='{}: precision'.format(blurs[i]),color=colors[i],linewidth=4)
    a.errorbar(thresh,n[:-1,0,1],yerr=n[:-1,1,1],label='{}: recall'.format(blurs[i]),color=colors[i],linestyle=':',linewidth=4)
#    a.errorbar(thresh,n[:-1,0,2],yerr=n[:-1,1,2],label='{}: F1'.format(blurs[i]),color=colors[i],linestyle='-.',linewidth=4)
a.legend(fontsize=20)
a.set_ylabel('Precision/Recall',fontsize=20)
a.set_xlabel('Threshold Value',fontsize=20)
f.savefig('/home/jay/data/model_comparison/precision_recall_2ConvNormlr0001.png')

# %%

n = error_stats[1,4]
thresh = np.arange(10)/10 + 0.1
colors = ['red','blue','black']
blurs = ['no blur','blur 1','blur 2']

f,a = plt.subplots(ncols=2,figsize=(30,20))

for i in range(3):
    n = error_stats[i,4]
    a[0].errorbar(thresh,n[:-1,0,0],yerr=n[:-1,1,0],label='{}: precision'.format(blurs[i]),color=colors[i],linewidth=4)
    a[0].errorbar(thresh,n[:-1,0,1],yerr=n[:-1,1,1],label='{}: recall'.format(blurs[i]),color=colors[i],linestyle=':',linewidth=4)
    a[1].errorbar(thresh,n[:-1,0,2],yerr=n[:-1,1,2],label='{}: F1'.format(blurs[i]),color=colors[i],linewidth=4)
a[0].legend(fontsize=20)
a[1].legend(fontsize=20)
a[0].set_ylabel('Precision/Recall',fontsize=20)
a[0].set_xlabel('Threshold Value',fontsize=20)
a[1].set_ylabel('F1 Score',fontsize=20)
a[1].set_xlabel('Threshold Value',fontsize=20)
a[1].set_ylim([0,1])

f.savefig('/home/jay/data/model_comparison/precision_recall_F1_NormOnlylr0001.png')

# %%

label_sizes = np.zeros((50,1))
for i in range(len(label)):
    print(i)
    lab = measure.label(label[i])
    s = []
    for region in measure.regionprops(lab):
        s.append(region.major_axis_length)
    label_sizes[i] = np.mean(s)
    
# %%

label_label = measure.label(label[0])
label_sizes = []
for region in measure.regionprops(label_label):
    label_sizes.append(region.major_axis_length*0.18)

# %%
    
particle_sizes = np.zeros((3,5,11,5))

for blur in range(3):
    print('blur: ',blur)
    for model in range(5):
        print('model: ',model)
        pred = total_results[blur,model,0]
        for thresh in range(11):
            print(thresh)
            if thresh == 10:
                t = filters.threshold_otsu(pred)
            else:
                t = np.round(thresh/10 + 0.1,2)
            im_label = measure.label(pred > t)
            s = []
            for region in measure.regionprops(im_label):
                s.append(region.major_axis_length*0.18)
            particle_sizes[blur,model,thresh,:] = [np.mean(s),np.std(s),np.mean(label_sizes),np.std(label_sizes),t]
            print(len(s))
            print('\n')
            
# %%
            
f,a = plt.subplots(figsize=(20,20))
blurs = ['no blur','blur 1','blur 2']
colors=['red','blue','black']
for i in range(3):
    data = particle_sizes[i,4,:]
    a.errorbar(data[:,-1],data[:,0],yerr=data[:,1],label='{}'.format(blurs[i]),linewidth=4,color=colors[i])
a.plot(data[:,-1],data[:,2],label='label',linewidth=4,color='green')
a.set_ylabel('Particle Size [nm]',fontsize=20)
a.set_xlabel('Threshold Value',fontsize=20)
a.set_ylim([0,25])
a.legend(fontsize=20)
f.savefig('/home/jay/data/model_comparison/size_v_threshold.png')

# %%

f,a = plt.subplots(ncols=2,figsize=(30,20))
a[0].imshow(raw_images[0],plt.cm.gray)
a[0].imshow(total_results[0,4,0],plt.cm.jet,alpha=0.5)

a[1].imshow(raw_images[0],plt.cm.gray)
a[1].imshow(total_results[0,4,0] > filters.threshold_otsu(total_results[0,4,0]),plt.cm.jet,alpha=0.5)

# %%

f,a = plt.subplots(figsize=(20,20))
a.imshow(raw_images[0],plt.cm.gray)
a.imshow(total_results[2,4,0],plt.cm.jet,alpha=0.5)

# %%
        
f,a = plt.subplots(figsize=(20,20))
colors=['red','blue','black']
blurs = ['no blur','blur 1','blur 2']

thresh = np.arange(10)/10 + 0.1

for i in range(3):
    n = all_sizes[i,4]
    a.errorbar(n[:,-1],n[:,0]*0.18,yerr=n[:,1],color=colors[i],label='{}'.format(blurs[i]),linewidth=4)
    
a.plot(thresh,n[:-1,2]*0.18,color='green',linestyle=':',linewidth=4)
a.set_ylabel('Particle Size [pixels]',fontsize=20)
a.set_ylim([0,50])
a.set_xlabel('Threshold Value',fontsize=20)
a.legend(fontsize=20)

# %%

f,a = plt.subplots(figsize=(20,20))
a.hist(s,color='red',label='predicted')
a.hist(s1,color='blue',label='label',alpha=0.5)
a.legend()

# %%
    
tensor_im = torch.Tensor(raw_images.reshape(50,1,1024,1024))
dataset = torch.utils.data.TensorDataset(tensor_im)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False)    

# %%

#r1 = np.zeros((50,2,1024,1024))
r1_soft = np.zeros((50,2,1024,1024))
checkpoint = '/home/jay/data/no_blur/norm_lr0001_NoDecay/norm_lr0001_NoDecay_e25.pt'
model = twoconv_norm()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ',device)
model.to(device)
model = nn.DataParallel(model)
    
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

n = 0
batch_size = 8

for i, data in enumerate(dataloader,0):
    print(i)
    inputs = data[0]
    p = model(inputs.cuda())
    p_soft = F.softmax(p)
    p = p.detach().cpu().numpy()
    p_soft = p_soft.detach().cpu().numpy()
#    r1[n*batch_size:n*batch_size+len(inputs)] = p
    r1_soft[n*batch_size:n*batch_size+len(inputs)] = p_soft
    n += 1
    
# %%
    
f,a = plt.subplots(ncols=2,figsize=(30,20))
a[0].imshow(raw_images[0],plt.cm.gray)
a[0].imshow(label[0],plt.cm.jet,alpha=0.3)
a[1].imshow(raw_images[0],plt.cm.gray)
a[1].imshow(pred0,plt.cm.jet,alpha=0.3)

# %%

error_stats = np.zeros((len(label),4))

for i in range(len(error_stats)):
    l = label[i].astype(bool).reshape(1024*1024,1)
    pred = (r1_soft[i,1] > 0.7).reshape(1024*1024,1)
    a = precision_recall_fscore_support(l,pred,average='binary')
    error_stats[i,:] = [a[0],a[1],a[2],a[3]]
    
np.save('/home/jay/data/model_comparison/norm_lr0001_25_prfs.npy',error_stats)
    