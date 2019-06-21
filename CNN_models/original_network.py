#!/usr/bin/env python
# coding: utf-8


# %% 




# In[1]:


import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as D
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder
print(torch.__version__)
import time

def predict_image(image):
    image = image.reshape(-1,1,image.shape[0],image.shape[0])
    #print(image.shape)
    image_tensor = torch.from_numpy(image)
    input = torch.autograd.Variable(image_tensor)
    input = input.to(device,dtype=torch.float)
    output = model(input)
    return output.cpu().detach().numpy()


# %%
#images = np.load(r'/home/jay/data/03072019/Heat4_images.npy')
batch_size = 8
    
    
images = np.load('/home/jay/data/no_blur/test_images.npy')[:50]
images = images.reshape(images.shape[0],1,images.shape[1],images.shape[2])
print(images.shape)
tensor_im = torch.stack([torch.Tensor(i) for i in images])
dataset = torch.utils.data.TensorDataset(tensor_im)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)

# In[3]:


# =============================================================================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,8,20,padding=(10,10))
#         self.conv2 = nn.Conv2d(8,16,10,padding=(5,5))
#         self.conv3 = nn.Conv2d(16,32,3,padding=(1,1))
#         self.conv4 = nn.Conv2d(32,32,3,padding=(1,1))
#         self.conv5 = nn.Conv2d(32,16,3,padding=(1,1))
#         self.conv6 = nn.Conv2d(16,8,3,padding=(1,1))
#         self.conv7 = nn.Conv2d(8,2,3,padding=(1,1))
#         self.conv8 = nn.Conv2d(2,2,1)#,padding=(1,1))
#         self.pool = nn.MaxPool2d(2,2)
#         self.upsample = nn.Upsample(scale_factor=2)
#     def forward(self,x):
#         x = F.relu(self.pool(self.conv1(x)))
#         x = F.relu(self.pool(self.conv2(x)))
#         x = F.relu(self.pool(self.conv3(x)))
#         x = F.relu(self.upsample(self.conv4(x)))
#         x = F.relu(self.upsample(self.conv5(x)))
#         x = F.relu(self.upsample(self.conv6(x)))
#         x = self.conv7(x)
#         x = self.conv8(x)
#         x = torch.reshape(x,(-1,2,512*512))
#         #x = nn.LogSoftmax(x)
#         return x
# net = Net()
# =============================================================================
        
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding=(1,1))
        self.conv2 = nn.Conv2d(8,16,3,padding=(1,1))
        self.conv3 = nn.Conv2d(16,32,3,padding=(1,1))
        self.conv4 = nn.ConvTranspose2d(32,16,3,stride=2,padding=(1,1))
        self.conv5 = nn.ConvTranspose2d(48,8,3,stride=2,padding=(1,1))
        self.conv6 = nn.ConvTranspose2d(24,8,3,stride=2,padding=(1,1))
        self.conv7 = nn.Conv2d(16,2,3,padding=(1,1))
        self.conv8 = nn.Conv2d(2,2,1)#,padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        #self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c1out = self.pool(c1)
        c2 = F.relu(self.conv2(c1out))
        c2out = self.pool(c2)
        c3 = F.relu(self.conv3(c2out))
        c3out = self.pool(c3)
        c4 = F.relu(self.conv4(c3out,output_size=c3.size()))
        c4cat = torch.cat((c4,c3),dim=1)
        c5 = F.relu(self.conv5(c4cat,output_size=c2.size()))
        c5cat = torch.cat((c5,c2),dim=1)
        c6 = F.relu(self.conv6(c5cat,output_size=c1.size()))
        c6cat = torch.cat((c6,c1),dim=1)
        c7 = F.relu(self.conv7(c6cat))
        
        return c7


model = UNet()
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model = nn.DataParallel(model)    

checkpoint = torch.load('/home/jay/ExtraDrive/1k_dataset/lr001_NoDecay_batch32_e200')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# %%
start = time.time()
predictions_soft = np.zeros((len(dataset),2,1024,1024))
raw = np.zeros((len(dataset),1,1024,1024))
n = 0

for i, data in enumerate(dataloader,0):
    print(i)
    inputs = data[0]
#    raw[n*batch_size:n*batch_size+len(inputs),:,:,:] = inputs.detach().cpu().numpy()
    p = model(inputs.cuda())
    #p = F.softmax(p)
    p = p.detach().cpu().numpy()
    predictions_soft[n*batch_size:n*batch_size+len(inputs),:,:,:] = p
    n += 1
    
print(time.time() - start)

# %%
        
f,a = plt.subplots(ncols=2,figsize=(40,20))

#n = 285
n = int(np.random.rand()*len(images))
print(n)

a[0].imshow(raw[n,0,:],plt.cm.gray)
a[0].imshow(predictions[n,0,:,:],plt.cm.jet,alpha=0.3)
a[1].imshow(raw[n,0,:],plt.cm.gray)
a[1].imshow(predictions[n,1,:,:],plt.cm.jet,alpha=0.3)

# %%

f,a = plt.subplots(figsize=(20,20))

n = 158
#n = int(np.random.rand()*len(images))
print(n)

a.imshow(raw[n,0,:],plt.cm.gray)
a.imshow(predictions[n,1,:,:],plt.cm.jet,alpha=0.3)

f.savefig('lr001_NoDecay_batch32_example.jpg')


# %%

plt.subplots(figsize=(20,20))
plt.imshow(test_image,plt.cm.gray)
plt.imshow(output[0,1],plt.cm.jet,alpha=0.3)
# %%

np.save('slowLR_predictions.npy',predictions)