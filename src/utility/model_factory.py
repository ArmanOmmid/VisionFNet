import __init__

import sys
import os

import time
from torch.utils.data import DataLoader
import torch
import gc
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
import numpy as np
from collections import Counter
import random
from matplotlib import pyplot as plt

import src.utility.util as util
import src.utility.voc as voc
import src.models.transfer_fcn as transfer_fcn
import src.models.unet as unet
import src.models.customfcn1 as customfcn1
import src.models.customfcn2 as customfcn2
from src.models.basic_fcn import *

MODE = ['lr', 'weight', 'custom1']
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

def init_weights(m):
    if 'transfer' in MODE:
        if isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases
    else:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

def getClassWeights(dataset, n_class):
    cum_counts = torch.zeros(n_class)
    for iter, (inputs, labels) in enumerate(train_loader_no_shuffle):
        labels = torch.squeeze(labels) # 224 x 224
        vals, counts = labels.unique(return_counts = True)
        for v, c in zip(vals, counts):
            cum_counts[v.item()] += c.item()
        #print(f"Cumulative counts at iter {iter}: {cum_counts}")
    totalPixels = torch.sum(cum_counts)
    classWeights = 1 - (cum_counts / totalPixels)
    print(f"Class weights: {classWeights}")
    return classWeights

if 'unet' in MODE:
    model = unet.UNet(n_class=n_class)
elif 'transfer' in MODE:
    model = transfer_fcn.Resnet_FCN(n_class=n_class)
elif 'custom1' in MODE:
    model = customfcn1.Custom_FCN1(n_class=n_class)
elif 'custom2' in MODE:
    model = customfcn2.Custom_FCN2(n_class=n_class)
else:
    model = FCN(n_class=n_class)


model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # determine which device to use (cuda or cpu)

if 'transfer' in MODE:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.Adam(params_to_update, lr = learning_rate, weight_decay=0.001)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

if 'lr' in MODE:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

if 'weight' in MODE:
    classWeights = getClassWeights(train_loader_no_shuffle, n_class).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights) 
else:
    criterion = nn.CrossEntropyLoss() 

model = model.to(device) # transfer the model to the device