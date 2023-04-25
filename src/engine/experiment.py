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

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

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

class Experiment(object):

    def __init__(self,
            model: torch.nn.Module, 
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            device: torch.device,
            model_save_path: str,
        ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_save_path = model_save_path
    
    def train(self, epochs, early_stop_tolerance):
        
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
            for iter, (inputs, labels) in enumerate(self.train_loader):
                # reset optimizer gradients
                self.optimizer.zero_grad()

                # both inputs and labels have to reside in the same device as the model's
                inputs =  inputs.to(self.device)# transfer the input to the same device as the model's
                labels =  labels.to(self.device) # transfer the labels to the same device as the model's
                
                if 'augment' in MODE:
                    # due to crop transform
                    b, ncrop, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    b, ncrop, h, w = labels.size()
                    labels = labels.view(-1, h, w)
                
                outputs = self.model(inputs) # Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

                loss = self.criterion(outputs, labels)  # calculate loss

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
                self.optimizer.step()

                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

            if 'lr' in MODE:
                print(f'Learning rate at epoch {epoch}: {self.scheduler.get_lr()[0]:0.9f}')  # changes every epoch
                # lr scheduler
                self.scheduler.step()           
                        
            with torch.no_grad():
                train_loss_at_epoch = np.mean(losses)
                train_iou_at_epoch = np.mean(mean_iou_scores)
                train_acc_at_epoch = np.mean(accuracy)

                train_loss_per_epoch.append(train_loss_at_epoch)
                train_iou_per_epoch.append(train_iou_at_epoch)
                train_acc_per_epoch.append(train_acc_at_epoch)

                print("Finishing epoch {}, time elapsed {}".format(epoch, time.time() - ts))

                valid_loss_at_epoch, valid_iou_at_epoch, valid_acc_at_epoch = self.val(epoch)

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
                    torch.save(self.model.state_dict(), self.model_save_path)
                else:
                    early_stop_count += 1
                    if early_stop_count > early_stop_tolerance:
                        print("Early Stopping...")
                        break
        self.model.load_state_dict(torch.load(self.model_save_path))
                
        return best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch
    
    def val(self, current_epoch):

        self.model.eval() # Put in eval mode (disables batchnorm/dropout) !
        
        losses = []
        mean_iou_scores = []
        accuracy = []

        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            for iter, (input, label) in enumerate(self.val_loader):
                input = input.to(self.device)
                label = label.to(self.device)
                
                output = self.model(input)
                loss = self.criterion(output, label)
                losses.append(loss.item())
                _, pred = torch.max(output, dim=1)
                acc = util.pixel_acc(pred, label)
                accuracy.append(acc)
                iou_score = util.iou(pred, label)
                mean_iou_scores.append(iou_score)
            loss_at_epoch = np.mean(losses)
            iou_at_epoch = np.mean(mean_iou_scores)
            acc_at_epoch = np.mean(accuracy)

        print(f"Valid Loss at epoch: {current_epoch} is {loss_at_epoch}")
        print(f"Valid IoU at epoch: {current_epoch} is {iou_at_epoch}")
        print(f"Valid Pixel acc at epoch: {current_epoch} is {acc_at_epoch}")

        self.model.train() # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

        return loss_at_epoch, iou_at_epoch, acc_at_epoch
    
    def test(self):

        self.model.eval()  # Put in eval mode (disables batchnorm/dropout) !

        losses = []
        mean_iou_scores = []
        accuracy = []

        with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

            for iter, (input, label) in enumerate(self.test_loader):
                input = input.to(self.device)
                label = label.to(self.device)

                output = self.model(input)
                loss = self.criterion(output, label)
                losses.append(loss.item())
                _, pred = torch.max(output, dim=1)
                acc = util.pixel_acc(pred, label)
                accuracy.append(acc)
                iou_score = util.iou(pred, label)
                mean_iou_scores.append(iou_score)

        test_loss = np.mean(losses)
        test_iou = np.mean(mean_iou_scores)
        test_acc = np.mean(accuracy)

        self.model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

        return test_loss, test_iou, test_acc
    