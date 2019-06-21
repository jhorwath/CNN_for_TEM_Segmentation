#!/usr/bin/env python
# coding: utf-8

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

save_name = '/home/jay/data/no_blur/norm_lr0001_OneStep/norm_lr0001_Decay10continue'
# In[2]:


#images = np.load('/home/jay/Desktop/training_set/training_im.npy')
#labels = np.load('/home/jay/Desktop/training_set/training_lab.npy')
images = np.load('/home/jay/data/train_sig2/test_images.npy')
#labels = np.load('/home/jay/data/no_blur/test_labels.npy')

#images = np.load('X_train.npy')
#labels = np.load('y_train_preprocessed.npy')
#print(len(images),len(labels))

# In[3]:


print(images.shape)
#print(labels.shape)
images = images.reshape(images.shape[0],1,1024,1024)
#labels = labels.reshape(labels.shape[0],1,1024,1024)
#labels = labels.reshape(labels.shape[0],512*512)
print(images.shape)
#print(labels.shape)


# In[4]:


tensor_im = torch.stack([torch.Tensor(i) for i in images])
#tensor_lab = torch.stack([torch.Tensor(i) for i in labels])
print(tensor_im.size())
dataset = torch.utils.data.TensorDataset(tensor_im)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False)


# %%

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
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

#class UNet(nn.Module):
#    def __init__(self):
#        super(UNet,self).__init__()
#        self.conv0 = nn.Conv2d(1,8,3,dilation=6,padding=6)
#        self.conv1 = nn.Conv2d(8,16,3,dilation=3,padding=3)
#        self.conv2 = nn.Conv2d(16,32,3,dilation=2,padding=2)
#        self.conv3 = nn.Conv2d(32,64,3,dilation=1,padding=1)
#        self.conv4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=(1,1))
#        self.conv5 = nn.ConvTranspose2d(96,32,3,stride=2,padding=(1,1))
#        self.conv6 = nn.ConvTranspose2d(64,16,3,stride=2,padding=(1,1))
#        self.conv7 = nn.ConvTranspose2d(32,16,3,stride=2,padding=(1,1))
#        self.conv8 = nn.Conv2d(24,8,3,padding=(1,1))
#        self.conv9 = nn.Conv2d(8,2,1)#,padding=(1,1))
#        self.pool = nn.MaxPool2d(2,2)
#        #self.upsample = nn.Upsample(scale_factor=2)
#        
#    def forward(self,x):
#        c0 = F.leaky_relu(self.conv0(x))
#        c0out = self.pool(c0)
#        c1 = F.leaky_relu(self.conv1(c0out))
#        c1out = self.pool(c1)
#        c2 = F.leaky_relu(self.conv2(c1out))
#        c2out = self.pool(c2)
#        c3 = F.leaky_relu(self.conv3(c2out))
#        c3out = self.pool(c3)
#        c4 = F.leaky_relu(self.conv4(c3out,output_size=c3.size()))
#        c4cat = torch.cat((c4,c3),dim=1)
#        c5 = F.leaky_relu(self.conv5(c4cat,output_size=c2.size()))
#        c5cat = torch.cat((c5,c2),dim=1)
#        c6 = F.leaky_relu(self.conv6(c5cat,output_size=c1.size()))
#        c6cat = torch.cat((c6,c1),dim=1)
#        c7 = F.leaky_relu(self.conv7(c6cat,output_size=c0.size()))
#        c7cat = torch.cat((c7,c0),dim=1)
#        c8 = F.leaky_relu(self.conv8(c7cat))
#        c9 = F.leaky_relu(self.conv9(c8))
#        
#        return c9



model = UNet()

criterion = nn.CrossEntropyLoss()
#1: lr001,with schedule gamma = 0.5 every 40steps
optimizer = optim.Adam(model.parameters(),lr = 0.0001) 
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model = nn.DataParallel(model)

checkpoint = torch.load('/home/jay/data/no_blur/norm_lr0001_OneStep/norm_lr0001_Decay10continue2_e205.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for state in optimizer.state.values():
    for k,v in state.items():
        if isinstance(v,torch.Tensor):
            state[k] = v.to(device)

#epoch = checkpoint['epoch']
#loss = checkpoint['loss']

#model.to(device)
#model = nn.DataParallel(model)
model.eval()

# %%

#for i,data in enumerate(dataloader,0):
#    if i > 0:
#        break
#    inputs,labels = data
#    inputs,labels = inputs.to(device),labels.to(device)
#    outputs = model(inputs).detach().cpu().numpy()
#    print(outputs.shape)
#    



# %%

loss_history = []
start = time.time()
for epoch in range(50):

    start_epoch = time.time()
    print('Epoch {}'.format(epoch+51))
    running_loss = 0.0
    n = 0
#    scheduler.step()
    for i, data in enumerate(dataloader,0):

        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
#        print(inputs.size(),labels.size())
      
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs,labels.long())
        loss.backward()        
        optimizer.step()
        
        
        running_loss += loss.item()
        if i % (len(dataloader)/10) == 0:
            print('{}%'.format(n))
            n += 10
    print('loss: {}'.format(running_loss/len(dataloader)))
    #print('loss: {}'.format(running_loss/1340))
    print('Epoch time: {}\n'.format(time.time() - start_epoch))
    loss_history.append(running_loss/len(dataloader))
    
    
    #loss_history.append(running_loss/1340)
    if (epoch + 1) % 5 == 0:
        f,a = plt.subplots(ncols=2,figsize=(30,20))
        im = inputs.detach().cpu().numpy()[0,0,:,:]
        out_raw = outputs.detach().cpu().numpy()[0,1,:,:]
        out = F.softmax(outputs).detach().cpu().numpy()[0,1,:,:]
        a[0].imshow(im,plt.cm.gray)
        a[0].imshow(out_raw,plt.cm.jet,alpha=0.3)
        a[0].set_title('Raw   Max={}'.format(np.max(out_raw)))
        a[1].imshow(im,plt.cm.gray)
        a[1].imshow(out,plt.cm.jet,alpha=0.3)
        a[1].set_title('Softmax  Max: {}'.format(out.max()))
        f.savefig(save_name+'_e{}.jpg'.format(epoch+51))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        },save_name+'_e{}.pt'.format(epoch+51))
print('Finished Training')
print('Total Time: {}'.format(time.time() - start))

f,a = plt.subplots(figsize=(10,10)) 
a.set_xlabel('Epoch') 
a.set_ylabel('Loss') 
#a.set_title('UNet 04032019') 
a.plot(range(len(loss_history)),loss_history) 
f.savefig(save_name+'_Loss.jpg')


## %%
#### Inference
#
#start = time.time()
#predictions50 = np.zeros((len(dataset),2,1024,1024))
##raw = np.zeros((len(dataset),1,1024,1024))
#n = 0
#batch_size=8
#for i, data in enumerate(dataloader,0):
#    print(i)
#    inputs = data[0]
##    raw[n*batch_size:n*batch_size+len(inputs),:,:,:] = inputs.detach().cpu().numpy()
#    p = model(inputs.cuda())
#    p = F.softmax(p)
#    p = p.detach().cpu().numpy()
#    predictions50[n*batch_size:n*batch_size+len(inputs),:,:,:] = p
#    n += 1
#    
#print(time.time() - start)
#
#
#
## %%
#
#f,a = plt.subplots(ncols=2,figsize=(40,30))
#n = int(np.random.rand()*len(images))
#a[0].imshow(images[n,0,:,:],plt.cm.gray)
#a[0].imshow(predictions5[n,1,:,:]>0.7,plt.cm.jet,alpha=0.3)
#a[1].imshow(images[n,0,:,:],plt.cm.gray)
#a[1].imshow(predictions50[n,1,:,:]>0.7,plt.cm.jet,alpha=0.3)


# %%

#f,a = plt.subplots(figsize=(20,20))
#ax = a.imshow(im,plt.cm.gray)
#ax = a.imshow(out,plt.cm.jet,alpha=0.3)
#f.colorbar(ax)
#f.savefig('/home/jay/data/no_blur/norm_lr0001_OneStep/output100.jpg')
#
#f,a = plt.subplots(figsize=(20,20))
#ax = a.imshow(im,plt.cm.gray)
#ax = a.imshow(out > 0.7,plt.cm.jet,alpha=0.3)
#f.savefig('/home/jay/data/no_blur/norm_lr0001_OneStep/output100_thresh.jpg')


# %%

start = time.time()
predictions = np.zeros((len(dataset),2,1024,1024))
batch_size=8
#raw = np.zeros((len(dataset),1,1024,1024))
n = 0

for i, data in enumerate(dataloader,0):
    print(i)
    inputs = data[0]
#    raw[n*batch_size:n*batch_size+len(inputs),:,:,:] = inputs.detach().cpu().numpy()
    p = model(inputs.cuda())
#    p = F.softmax(p)
    p = p.detach().cpu().numpy()
    predictions[n*batch_size:n*batch_size+len(inputs),:,:,:] = p
    n += 1
    
print(time.time() - start)
np.save('/home/jay/data/2blurred_images_no_blur_network_raw.npy',predictions)

# %%

n = int(np.random.rand()*len(images))
f,a = plt.subplots(ncols=2,figsize=(30,20))
a[0].imshow(images[n,:,:],plt.cm.gray)
a[0].imshow(p_raw[n,1,:,:],plt.cm.jet,alpha=0.3)
a[1].imshow(images[n,:,:],plt.cm.gray)
a[1].imshow(p_soft[n,1,:,:] > 0.7,plt.cm.jet,alpha=0.3)