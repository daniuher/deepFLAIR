#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:34:23 2022

@author: g10060008
"""

import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


device = torch.device("cpu")
torch.set_num_threads(8)


class deepFLAIRdata():
    
    def __init__(self, patient):   
        
        self.patpath = os.path.join(os.getcwd(), patient)            
        self.mask = []       
        self.seqs = [x for x in os.listdir(self.patpath) if x.startswith('c') and x.endswith('.nii')] 
      
        
    def load(self, selected_sequences):
        
        filepaths = []
        
        for seq in selected_sequences:                        
            if seq in self.seqs:                
                filepaths.append(os.path.join(self.patpath, seq))
            else:
                print(f'Very sorry, but sequence {seq} cannot be found.')
        
        self.imgs = [nib.load(impath) for impath in filepaths]
        self.affine = self.imgs[0].affine
        self.data_shape = self.imgs[0].get_fdata().shape
        
        try:
            self.mask = nib.load(os.path.join(self.patpath, 'mask.nii'))
        except:
            print("Could not find a mask.nii file.")
        
     
    def get_brains(self):        
        self.brains = [np.multiply(x.get_fdata(), self.mask.get_fdata()) for x in self.imgs]    
    
    def set_train_indices(self, train_size):
        self.train_inds = []
    
    def get_voxel_surroundings(self, pos, size=1):
        
        surround_inds = []
        pp = np.linspace(-size,size,num=np.diff([-size,size])[0]+1, dtype=int) #possible dimensional positions
        for i in range(len(pp)):
            for j in range(len(pp)):
                for k in range(len(pp)):
                    surround_inds.append([pos[0]+pp[i], pos[1]+pp[j], pos[2]+pp[k]])
                    
        
        return surround_inds
    
    def get_random_index(self, up_lim):
        
        ind = np.random.randint(0,up_lim)
        c = True
        while c==True:
            if ind in self.train_inds:
                ind = np.random.randint(0,up_lim)
            else:
                self.train_inds.append(ind)
                c=False               
                return ind
            
     
    def train_test_split(self, brains=False, nonzero=False):
                
        train_slices = np.concatenate([np.linspace(0, 99,100), np.linspace(190, 319, 130)]).astype(int)
        trainX = pd.DataFrame()
        trainY = pd.DataFrame()       
        testX = pd.DataFrame()
        testY = pd.DataFrame()             
        
        if brains:
            
            trainY['y'] = self.brains[0][:,:,train_slices].flatten() # selecting spaceflair training labels    
            testY['y'] = self.brains[0].flatten() #getting the voxels of spaceflair as labels
            
            # trainY['y1'] = self.brains[0][:,:,train_slices].flatten() # selecting spaceflair training labels    
            # testY['y1'] = self.brains[0].flatten() #getting the voxels of spaceflair as labels
            
            # trainY['y2'] = self.brains[1][:,:,train_slices].flatten() # selecting spaceflair training labels    
            # testY['y2'] = self.brains[1].flatten() #getting the voxels of spaceflair as labels
            
            images = self.brains[1:]
            
            i = 0
            for brain in images:
                
                # trainX extraction - top 100 slices of the brain
                trainX[i] = brain[:,:,train_slices].flatten()
    
                # testX
                testX[i] = brain.flatten()
                
                i += 1
            
        else:    
            
            trainY['y'] = self.imgs[0].get_fdata()[:,:,train_slices].flatten() # selecting spaceflair training labels    
            testY['y'] = self.imgs[0].get_fdata().flatten() #getting the voxels of spaceflair as labels
            
            images = self.imgs[1:]
                    
            i = 0
            for image in images:
                
                im = image.get_fdata()
                
                # trainX extraction - top 100 slices of the brain
                trainX[i] = im[:,:,train_slices].flatten()
    
                # testX
                testX[i] = im.flatten()
                
                i += 1
        
        scaler = preprocessing.MinMaxScaler().fit(trainX)
        trainX = scaler.transform(trainX)
        
        scaler = preprocessing.MinMaxScaler().fit(trainY)
        trainY = scaler.transform(trainY)
        
        scaler = preprocessing.MinMaxScaler().fit(testX)
        testX = scaler.transform(testX)
        
        scaler = preprocessing.MinMaxScaler().fit(testY)
        testY = scaler.transform(testY)
        
        if nonzero:
            nztrainX = pd.DataFrame()
            nztrainY = pd.DataFrame()
            
            inds = np.nonzero(trainY)[0]
            
            nztrainX = trainX[inds, :]
            nztrainY = trainY[inds]
            
            return nztrainX, nztrainY, testX, testY
        
        else:
        
            return trainX, trainY, testX, testY
        
        

class dFmodel(nn.Module):
    def __init__(self, input_shape):
        super(dFmodel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, X):        
        return self.layers(X)
    
    
selected = ['cSPCFLR2.nii', 
            'cT2w.nii', 'cINV1.nii', 'cINV2.nii', 
            'cGREe1.nii', 'cGREe2.nii', 'cGREe3.nii', 'cGREe4.nii']


data = deepFLAIRdata('sub_xxx')
data.load(selected)
data.get_brains()
trainX, trainY, testX, testY = data.train_test_split(brains=True, nonzero=True)
inds = np.arange(0,trainX.shape[0],1)
np.random.shuffle(inds)

inds = data.get_voxel_surroundings([20,10,32])



model = dFmodel(len(selected)-1).to(device)
print(model)


loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

n_epochs = 10
n_iter = trainX.shape[0]

print('deepFLAIR started ...')
for epoch in range(n_epochs):
    mess = 'Epoch '+str(epoch+1)+'/'+str(n_epochs)
    
    # data.set_train_indices(trainX.shape[0])
    for i in tqdm(range(n_iter), desc=mess):
        
        # ind = data.get_random_index(trainX.shape[0])
        
        inputs = torch.tensor(trainX[inds[i],:], dtype=torch.float32, requires_grad=True)
        target = torch.tensor(trainY[inds[i]], dtype=torch.float32, requires_grad=True)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        output = model(inputs)
        
        # calculate FLAIR
        # flair = 1 - (2*np.exp(-TI/output[1].item()) + (np.exp(-TR/output[2].item()))    #1 - (2*exp(-TI/T1pix)) + (exp(-TR/T1pix));
        # sub_target = torch.tensor()
                     
        # Compute loss
        l = loss(output, target)
        
        # Perform backward pass
        l.backward()
      
        # Perform optimization
        optimizer.step()
        
        
    # Print statistics
    print(f'Epoch {epoch+1}/{n_epochs}: Loss: {l.item()}')
              

out = np.zeros((testX.shape[0], testX.shape[1]+3))
# out = np.zeros(testX.shape)   
# with torch.no_grad():

# https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb 
for param in model.parameters():
    param.requires_grad=False

# set model to evaluate mode
model.eval()
    
   
for i in tqdm(range(testX.shape[0]), desc="Evaluating"):
    
    if testY[i,0] != 0:
        
        inputs = torch.tensor(testX[i,:], dtype=torch.float32, requires_grad=True)
        target = torch.tensor(testY[i], dtype=torch.float32)
        # print(inputs)
    
    
        output = model(inputs)
        # print(output)
        
        l = loss(output, target)
        l.backward()
        
        grads = inputs.grad
        # print(grads)
        grads = torch.abs(grads.unsqueeze(0))
        
        
        for j in range(out.shape[1]-3):
            out[i,j] = grads[0][j].item()
        
        if grads.sum().item() == 0:
            out[i,-2] = 0
            out[i,-3] = 0
        else:    
            _, pos = torch.max(grads, 1)
            out[i,-2] = pos.item()+1    
            out[i,-3] = grads[0][pos.item()] / grads.sum().item()
        
        out[i,-1] = output.item()
    

for k in range(out.shape[1]):
    outim = out[:,k]
    outim = np.reshape(outim, data.data_shape)        
    img = nib.Nifti1Image(outim, data.affine)
    img.header.set_xyzt_units(xyz='mm', t='sec')
    nib.save(img, os.path.join(data.patpath, 'out'+str(k+1)+'.nii'))
        

    
    
    
    
    

    
        
    
    
    
    
    
