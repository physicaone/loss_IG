import numpy as np
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
from datetime import datetime
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")
from energy_rbm import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    device='cuda'
else:
    device='cpu'
torch.cuda.is_available()
n_hid=4
model0=torch.load('ERBM,SGD/lr=0.01/models/2021-01-28_n_hid={n_hid}_vol=12000_epoch=5000'.format(n_hid=n_hid), map_location=device).float()

def binary_to_decimal(list0):
    value=0
    for i in range(len(list0)):
        value+=list0[-i-1]*2**(i)
    return value

def decimal_to_binary(integer):
    string=bin(integer)[2:]
    list0=[float(d) for d in string]
    while len(list0)<n_hid:
        list0=[0.]+list0
    return torch.tensor([list0])

def check_energy(model0, n_sample=10000, step_eq=1):
    n_vis=len(model0.v[0]); n_hid=len(model0.h[0])
    state_E_dict={}
    hidden_state_list=[]
    for i in range(2**n_hid):
        # state_E_dict[str(i)]=[]
        state_E_dict[str(i)]=0
    # for i in range(2**n_hid):
    for i in [0]:
        hidden_state_list.append(decimal_to_binary(i))
    # for i in tqdm(range(len(hidden_state_list))):
    for i in range(len(hidden_state_list)):
        h_sample=hidden_state_list[i]
        for j in range(step_eq):
            v_sample=model0.hidden_to_visible(h_sample)
            v_sample=v_sample.bernoulli()
            h_sample=model0.visible_to_hidden(v_sample)
            h_sample=h_sample.bernoulli()
        v_sample=model0.hidden_to_visible(h_sample)
        v_sample=v_sample.bernoulli()
        for n in tqdm(range(n_sample), desc=bin(i)[2:]):
        # for n in range(n_sample):
            h_sample=model0.visible_to_hidden(v_sample)
            h_sample=h_sample.bernoulli()
            v_sample=model0.hidden_to_visible(h_sample)
            v_sample=v_sample.bernoulli()
            # state_E_dict[str(binary_to_decimal(h_sample.view(4).int().detach().numpy()))].append(model0.energy2(v_sample, h_sample).detach().numpy()[0][0].astype(int))
            state_E_dict[str(binary_to_decimal(h_sample.view(4).int().detach().numpy()))]+=1
    with open('state_E_dict_n_hid={n_hid}.pkl'.format(n_hid=n_hid), 'wb') as f:
        pkl.dump(state_E_dict, f)


check_energy(model0, n_sample=100000000, step_eq=1)