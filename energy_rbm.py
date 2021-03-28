
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.utils.data
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pandas as pd
import random
import powerlaw as pl
np.seterr(divide='ignore', invalid='ignore')
import pickle as pkl

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True

class RBM(nn.Module):

    def __init__(self, n_vis, n_hid, k, use_cuda):
        """Create a RBM."""
        super(RBM, self).__init__()
        
        if use_cuda==True:
            self.v = nn.Parameter(torch.ones(1, n_vis).cuda())
            self.h = nn.Parameter(torch.zeros(1, n_hid).cuda())
            self.W = nn.Parameter(torch.randn(n_hid, n_vis).cuda())
            self.k = k
        else:
            self.v = nn.Parameter(torch.ones(1, n_vis))
            self.h = nn.Parameter(torch.zeros(1, n_hid))
            self.W = nn.Parameter(torch.randn(n_hid, n_vis))
            self.k = k            

    def visible_to_hidden(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.h))

    def hidden_to_visible(self, h):
        return torch.sigmoid(F.linear(h, self.W.t(), self.v))

    def free_energy(self, v):
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)
    
    def energy(self, v):
        v=v.bernoulli()
        h=torch.sigmoid(F.linear(v, self.W, self.h))
        h=h.bernoulli()
        return -torch.matmul(v, self.v.t())-torch.matmul(torch.matmul(v, self.W.t()),h.t())-torch.matmul(h, self.h.t())

    def forward(self, v):
        h = self.visible_to_hidden(v)
        h = h.bernoulli()
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            v_gibb = v_gibb.bernoulli()
            h = self.visible_to_hidden(v_gibb)
            h = h.bernoulli()
        return v, v_gibb

def transform_str_list(list0):
    list1=[]
    for i in range(len(list0)):    
        aa=np.array(list0[i].strip('][ tensor()').strip('\n').replace('\n','').split(', ')).astype(float)
        list1.append(aa)
    
    return torch.from_numpy(np.array(list1)).float()  

def get_first_Trues(idx, vol):
    i=0
    for k in range(len(idx)):
        if idx[k]==True:
            i=i+1
            if i>vol:
                idx[k:]=False
                break        
    return idx

def get_small_mnist(vol, train_or_test, class_list):
    if train_or_test=='train':
        dataset = datasets.MNIST(root='../dataset/MNIST',train=True,download=True,
                                 transform=transforms.ToTensor())
    elif train_or_test=='test':
        dataset = datasets.MNIST(root='../dataset/MNIST',train=False,download=True,
                                 transform=transforms.ToTensor())
        
    dataset.targets = torch.tensor(dataset.targets)
#     idx = dataset.targets==0

    idx_list=[]
    for k in class_list:
        idx = dataset.targets==k
        idx = get_first_Trues(idx, vol)
        idx_list.append(idx)
    idx=sum(idx_list)
    idx2=[]
    for i in range(len(idx)):
        if idx[i]>=1:
            idx2.append(True)
        else:
            idx2.append(False)
    idx2=torch.tensor(idx2)
    dataset.targets= dataset.targets[idx2]
    dataset.data = dataset.data[idx2]
    return dataset

def IG_loss(model0, model1, data0, v_sample0):
    E0=0; E1=0
    F_V = model0.free_energy(data0)

    v_sample0=torch.tensor(v_sample0)
    for i in range(len(v_sample0)):
        E0 += model0.energy(v_sample0[i])/len(v_sample0)
        E1 += model1.energy(v_sample0[i])/len(v_sample0)
    return float(F_V), float(E1), float(E0)

def check_energy(model0, model1, n_sample=10000, step_eq=500):
    E0_list=[]; E1_list=[]
    v_list=[]
    input_random=torch.round(torch.rand(n_vis).view(1, n_vis)).to(device)
    for i in range(step_eq):
        h_sample=model0.visible_to_hidden(input_random)
        h_sample=h_sample.bernoulli()
        v_sample=model0.hidden_to_visible(h_sample)
        v_sample=v_sample.bernoulli()
        input_random=v_sample
    for n in tqdm(range(n_sample)):
        h_sample=model0.visible_to_hidden(v_sample)
        h_sample=h_sample.bernoulli()
        v_sample=model0.hidden_to_visible(h_sample)
        v_list.append(np.squeeze(v_sample.detach().to(device).numpy()))
        v_sample=v_sample.bernoulli()
        E0_list.append(np.squeeze(model0.energy(v_sample).detach().to(device).numpy()))
        E1_list.append(np.squeeze(model1.energy(v_sample).detach().to(device).numpy()))
    return np.array(E0_list), np.array(E1_list), np.array(E1_list)-np.array(E0_list), v_list

