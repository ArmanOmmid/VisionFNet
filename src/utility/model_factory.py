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

def init_weights(module, decoder_only):
    if decoder_only:
        if isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(module.weight.data)
            torch.nn.init.normal_(module.bias.data) #xavier not applicable for biases
    else:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(module.weight.data)
            torch.nn.init.normal_(module.bias.data) #xavier not applicable for biases

def build_model(name, class_count):

    if name == 'unet':
        model = arch.unet.UNet(n_class=class_count)
    elif name == 'transfer':
        model = arch.transfer_fcn.Resnet_FCN(n_class=class_count)
    elif name == 'custom1':
        model = arch.customfcn1.Custom_FCN1(n_class=class_count)
    elif name == 'custom2':
        model = arch.customfcn2.Custom_FCN2(n_class=class_count)
    else:
        model = arch.basic_fcn.FCN(n_class=class_count)

    return model
