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

class Experiment(object):

    def __init__(self) -> None:
        pass

    def train():
        best_iou_score = 0.0
        best_loss = 100.0
        early_stop_count = 0
        train_loss_per_epoch = []
        train_iou_per_epoch = []
        train_acc_per_epoch = []
        
        valid_loss_per_epoch = []
        valid_iou_per_epoch = []
        valid_acc_per_epoch = []
        
        for epoch in range(epochs):
            ts = time.time()
            losses = []
            mean_iou_scores = []
            accuracy = []
            for iter, (inputs, labels) in enumerate(train_loader):
                # reset optimizer gradients
                optimizer.zero_grad()

                # both inputs and labels have to reside in the same device as the model's
                inputs =  inputs.to(device)# transfer the input to the same device as the model's
                labels =  labels.to(device) # transfer the labels to the same device as the model's
                
                if 'augment' in MODE:
                    # due to crop transform
                    b, ncrop, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    b, ncrop, h, w = labels.size()
                    labels = labels.view(-1, h, w)
                
                outputs = model(inputs) # Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

                loss = criterion(outputs, labels)  # calculate loss

                with torch.no_grad():
                    losses.append(loss.item())
                    _, pred = torch.max(outputs, dim=1)
                    acc = util.pixel_acc(pred, labels)
                    accuracy.append(acc)
                    iou_score = util.iou(pred, labels)
                    mean_iou_scores.append(iou_score)
                    
                # backpropagate
                loss.backward()

                # update the weights
                optimizer.step()

                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

            if 'lr' in MODE:
                print(f'Learning rate at epoch {epoch}: {scheduler.get_lr()[0]:0.9f}')  # changes every epoch
                # lr scheduler
                scheduler.step()           
                        
            with torch.no_grad():
                train_loss_at_epoch = np.mean(losses)
                train_iou_at_epoch = np.mean(mean_iou_scores)
                train_acc_at_epoch = np.mean(accuracy)

                train_loss_per_epoch.append(train_loss_at_epoch)
                train_iou_per_epoch.append(train_iou_at_epoch)
                train_acc_per_epoch.append(train_acc_at_epoch)

                print("Finishing epoch {}, time elapsed {}".format(epoch, time.time() - ts))

                valid_loss_at_epoch, valid_iou_at_epoch, valid_acc_at_epoch = val(epoch)
                valid_loss_per_epoch.append(valid_loss_at_epoch)
                valid_iou_per_epoch.append(valid_iou_at_epoch)
                valid_acc_per_epoch.append(valid_acc_at_epoch)

                if valid_iou_at_epoch > best_iou_score:
                    best_iou_score = valid_iou_at_epoch
                    # save the best model
                if valid_loss_at_epoch < best_loss:
                    print(f"Valid Loss {valid_loss_at_epoch} < Best Loss {best_loss}. (Valid IOU {valid_iou_at_epoch}) Saving Model...")
                    best_loss = valid_loss_at_epoch
                    early_stop_count = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    early_stop_count += 1
                    if early_stop_count > early_stop_tolerance:
                        print("Early Stopping...")
                        break
        model.load_state_dict(torch.load(model_save_path))
                
        return best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch