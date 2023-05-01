import __init__

import sys
import os
from collections import Counter

import torch
import torch.nn as nn

import src.utility.util as util
import src.arch as arch

def get_weight_initializer(transfer):
    def init_weights(module):
        if transfer:
            if isinstance(module, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(module.weight.data)
                torch.nn.init.normal_(module.bias.data) #xavier not applicable for biases
        else:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(module.weight.data)
                torch.nn.init.normal_(module.bias.data) #xavier not applicable for biases
    return init_weights

def build_model(architecture, classes, augment):

    class_count = classes if isinstance(classes, int) else len(classes)

    attr = util.Attributes(
        transfer = False,
        augment = augment
    )

    if architecture == 'unet':
        model = arch.unet.UNet(n_class=class_count)

    elif architecture == 'transfer':
        model = arch.transfer_fcn.Resnet_FCN(n_class=class_count)
        attr.transfer = True

    elif architecture == 'custom1':
        model = arch.customfcn1.Custom_FCN1(n_class=class_count)

    elif architecture == 'custom2':
        model = arch.customfcn2.Custom_FCN2(n_class=class_count)

    else:
        model = arch.basic_fcn.FCN(n_class=class_count)

    attr(model)

    init_weights = get_weight_initializer(model.transfer)
    model.apply(init_weights)

    return model
