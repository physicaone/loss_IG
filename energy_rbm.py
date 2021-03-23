
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

def get_listmk(epoch1):
    config_count={} # 각 hidden layer state 갯수 파악 (k)
    for i in range(len(epoch1)):
        config_count[epoch1[i]]=0
    for i in range(len(epoch1)):
        config_count[epoch1[i]]+=1
        
    listk=[]
    for i in range(len(list(config_count.values()))):
        listk.append(int(list(config_count.values())[i]))
    listmk=[]
    kcount={}


    # 갯수의 갯수 파악 (m_k)
    for i in range(len(listk)):
        kcount[listk[i]]=0
    for i in range(len(listk)):
        kcount[listk[i]]+=1
    for i in range(len(kcount)):
        listmk.append(kcount[sorted(list(kcount))[i]])

    return sorted(list(kcount)), listmk
    


def get_H_k(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    H_k=0
    for i in range(len(x)):
        H_k-=(x[i]*y[i]/N)*np.log2(x[i]*y[i]/N)
    return H_k

def get_H_s(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    H_s=0
    for i in range(len(x)):
        H_s-=(x[i]*y[i]/N)*np.log2(x[i]/N)
    return H_s

def get_mu_larger_than_1(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    mu=1+np.log(2)/(np.log2(N)-get_H_s(x,y))
    return mu

def get_mu_smaller_than_1(x, y):
    mu=1-1/(get_H_s(x,y)*np.log(2))
    return mu

def get_entropies(x,y):
    H_k=get_H_k(x,y)
    H_s=get_H_s(x,y)
    mus=get_mu_larger_than_1(x,y)
    mul=get_mu_smaller_than_1(x,y)
    print("H_k = %f, H_s = %f, mu = %f and %f" %(H_k, H_s, mus, mul))

def get_ccdf_y(x,y):
    listkmk=[]
    listkmk_cum=[]
    for i in range(len(y)):
        listkmk.append(x[i]*y[i])
    sum_listkmk=sum(listkmk)
    for i in range(len(listkmk)):
        listkmk_cum.append(sum(listkmk)/sum_listkmk)
        listkmk.pop(0)
    return listkmk_cum

def get_exponent(x,y):
    expo=-1-get_mu_larger_than_1(x,y)
    return expo

def get_state(x,y):
    list00=[]
    for i in range(len(x)):
        for j in range(y[i]):
            list00.append(x[i])

    return list00


def get_logbin(x,y):
    bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    list00=get_state(x,y)
    y1,x1,_ = plt.hist(list00, bins = bins, histtype='step', color='white')
    x1 = 0.5*(x1[1:]+x1[:-1])
    y1_=[]
    dummy=0
    for i in range(len(y1)):
        if y1[i]!=0.0:
            dummy+=1
    for i in range(dummy):
        y1_.append(y1[i]/((bins[i+1]-bins[i])))
    plt.close()

    return x1[0:dummy], y1_

def get_MLE_exponent(x,y):
    fit1 = pl.Fit(get_state(x, y), discrete=True, xmin=0)
    print(fit1.power_law.alpha)

def get_MLE_exponent_mk_cut(x,y):
    x_bin, y_bin=get_logbin(x,y)
    cri1=0
    cri2=0
    for i in range(len(y_bin)):
        if y_bin[i]<1.:
            cri1=i
            break
    x_cri=x_bin[cri1-1]
    for j in range(len(x)):
        if x[j]>x_cri:
            cri2=j
            break
    x2=x[0:cri2-1]; y2=y[0:cri2-1]
    fit1 = pl.Fit(get_state(x2, y2), discrete=True, xmin=0)
    print(fit1.power_law.alpha)

    return x2,y2








