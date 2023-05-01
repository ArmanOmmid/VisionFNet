import __init__

import sys
import os
import time
import gc
import random
from collections import Counter
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF

import numpy as np
from matplotlib import pyplot as plt

import src.utility.util as util
import src.utility.voc as voc
import src.arch as arch


MODE = ['lr', 'weight', 'custom1']

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])

    if 'augment' in MODE:
        images = list(TF.ten_crop(image, 128))
        masks = list(TF.ten_crop(mask, 128))
        for i in range(10):
            angles = [30, 60]
            for angle in angles:
                msk = masks[i].unsqueeze(0)
                img = TF.rotate(images[i], angle)
                msk = TF.rotate(msk, angle)
                msk = msk.squeeze(0)
                images.append(img)
                masks.append(msk)
                
        image = torch.stack([img for img in images])
        mask = torch.stack([msk for msk in masks])
        
    return image, mask

def valtest_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])
    
    return image, mask

def sample_transform(image, mask):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

def prepare_dataset(voc_root):

    train_dataset = voc.VOC(voc_root, 'train', transforms=train_transform)
    val_dataset = voc.VOC(voc_root, 'val', transforms=valtest_transform)
    test_dataset = voc.VOC(voc_root, 'test', transforms=valtest_transform)

    train_loader_no_shuffle = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    if 'augment' in MODE:
        train_loader = DataLoader(dataset=train_dataset, batch_size= 8, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size= 8, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size= 8, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

    return train_loader, val_loader, test_loader, train_loader_no_shuffle

def getClassWeights(dataset, n_class):
    cum_counts = torch.zeros(n_class)
    for iter, (inputs, labels) in enumerate(dataset):
        labels = torch.squeeze(labels) # 224 x 224
        vals, counts = labels.unique(return_counts = True)
        for v, c in zip(vals, counts):
            cum_counts[v.item()] += c.item()
        #print(f"Cumulative counts at iter {iter}: {cum_counts}")
    totalPixels = torch.sum(cum_counts)
    classWeights = 1 - (cum_counts / totalPixels)
    print(f"Class weights: {classWeights}")
    return classWeights