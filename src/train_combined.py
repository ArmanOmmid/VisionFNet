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
    
    return image, mask

voc_root = os.path.join(__init__.repository_root, "datasets", "VOC")

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

epochs = 20
n_class = 21
learning_rate = 0.01
early_stop_tolerance = 8
model_save_path = 'model.pth'

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

