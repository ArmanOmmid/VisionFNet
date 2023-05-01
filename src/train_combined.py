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

from src.engine.experiment import Experiment
from src.utility.data_factory import prepare_dataset, sample_transform, getClassWeights
from src.utility.model_factory import init_weights, build_model

MODE = ['lr', 'weight', 'custom1']
"""
None: baseline
'lr': 4a (lr schedule)
'augment': 4b (data augment)
'weight': 4c (weight)
'custom1': 5a-1 (custom1)
'custom2': 5a-2 (custom2)
'transfer': 5b (transfer)
'unet': 5c (unet)
"""

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

epochs = 20
class_count = 21
learning_rate = 0.01
early_stop_tolerance = 8

model_save_path = os.path.join(__init__.repository_root, "weights", "model.pth")
voc_root = os.path.join(__init__.repository_root, "datasets", "VOC")

train_loader, val_loader, test_loader, train_loader_no_shuffle = prepare_dataset(voc_root)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

model = build_model(MODE, class_count)
    

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
    classWeights = getClassWeights(train_loader_no_shuffle, class_count).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights)
else:
    criterion = nn.CrossEntropyLoss() 

model = model.to(device) # transfer the model to the device


if __name__ == "__main__":

    print("Initializing Experiments")

    experiment = Experiment(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        MODE,
        model_save_path
    )

    # experiment.val(0)  # show the accuracy before training
    
    print("Training")

    results = experiment.train(epochs, early_stop_tolerance)
    
    best_iou_score, \
    train_loss_per_epoch, \
    train_iou_per_epoch, \
    train_acc_per_epoch, \
    valid_loss_per_epoch, \
    valid_iou_per_epoch, \
    valid_acc_per_epoch = results
    
    print(f"Best IoU score: {best_iou_score}")
    util.plot_train_valid(train_loss_per_epoch, valid_loss_per_epoch, name='Loss')
    util.plot_train_valid(train_acc_per_epoch, valid_acc_per_epoch, name='Accuracy')
    util.plot_train_valid(train_iou_per_epoch, valid_iou_per_epoch, name='Intersection over Union')
    
    test_loss, test_iou, test_acc = experiment.test()
    print(f"Test Loss is {test_loss}")
    print(f"Test IoU is {test_iou}")
    print(f"Test Pixel acc is {test_acc}")
    
    # ------ GET SAMPLE IMAGE FOR REPORT -------
    test_sample_dataset = voc.VOC(voc_root, 'test', transforms=sample_transform)
    test_sample_loader = DataLoader(dataset=test_sample_dataset, batch_size=1, shuffle=False)
    model.eval()
    # untransformed original image
    orig_inp, _ = next(iter(test_sample_loader))
    
    # transformed image for input to network
    inp, label = next(iter(test_loader))
    inp = inp.to(device)
    label = label.to(device)
    output = model(inp)
    _, pred = torch.max(output, dim=1)

    util.save_sample(np.array(orig_inp[0].cpu(), dtype=np.uint8), label[0].cpu(), pred[0].cpu())
    model.train()
    # -------------------------------------------
    
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()

