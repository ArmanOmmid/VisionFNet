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

def build_model(name, class_count):

    if 'unet' in MODE:
        model = arch.unet.UNet(n_class=class_count)
    elif 'transfer' in MODE:
        model = arch.transfer_fcn.Resnet_FCN(n_class=class_count)
    elif 'custom1' in MODE:
        model = arch.customfcn1.Custom_FCN1(n_class=class_count)
    elif 'custom2' in MODE:
        model = arch.customfcn2.Custom_FCN2(n_class=class_count)
    else:
        model = arch.FCN(n_class=class_count)

    return model
