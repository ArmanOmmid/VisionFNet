import __init__

import sys
import os
import time
import gc
import random
from collections import Counter
import argparse
import copy

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

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from src.utility.data_info import get_task_type
from src.utility.config import Config

class Experiment(object):

    def __init__(self,
            config: Config,
            model: torch.nn.Module, 
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            device: torch.device,
            save_path = False,
            load_path = False
        ) -> None:

        self.config = config

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.task = get_task_type(self.train_loader)
        self.classification = bool(self.task == 'classification')
        self.segmentation = bool(self.task == 'segmentation')

        self.data_loaders = {
            'train': train_loader,
            'val' : val_loader,
            'test' : test_loader
        }
        self.dataset_sizes = {mode: len(loader.dataset) for mode, loader in self.data_loaders.items()}

        self.criterion = criterion

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.save_path = save_path
        self.load_path = load_path

        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))

        self.best_model_weights = None
    
    def run(self, num_epochs, early_stop_tolerance):
        
        early_stop_count = 0

        best_loss = 100.0
        best_accuracy = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        train_loss_per_epoch = []
        train_acc_per_epoch = []
        
        valid_loss_per_epoch = []
        valid_acc_per_epoch = []

        if self.segmentation:
            best_iou_score = 0.0
            train_iou_per_epoch = []
            valid_iou_per_epoch = []
        else:
            best_iou_score = train_iou_per_epoch = valid_iou_per_epoch = None
        
        for epoch in range(num_epochs):
            print('')
            print('-' * 20)
            print("Epoch {} / {}".format(epoch+1, num_epochs))
            print('-' * 20)

            if self.scheduler is not None:
                print("Learning Rate: {}".format(self.scheduler.get_last_lr()[0]))

            ts = time.time()

            train_loss_at_epoch, train_acc_at_epoch, train_iou_at_epoch = self.train(epoch+1)

            if self.scheduler is not None:
                self.scheduler.step()

            valid_loss_at_epoch, valid_acc_at_epoch, valid_iou_at_epoch = self.val(epoch+1)

            print("Epoch {} | Time Elapsed: {} |".format(epoch+1, (time.time() - ts)))
            print("                      Train | Accuracy: {} | Loss: {}".format(f"{train_acc_at_epoch:.4f}", f"{train_loss_at_epoch:.4f}"))
            print("                 Validation | Accuracy: {} | Loss: {}".format(f"{valid_acc_at_epoch:.4f}", f"{valid_loss_at_epoch:.4f}"))
            
            # Append results 
            train_loss_per_epoch.append(train_loss_at_epoch)
            train_acc_per_epoch.append(train_acc_at_epoch)
            if self.segmentation:
                train_iou_per_epoch.append(train_iou_at_epoch)

            valid_loss_per_epoch.append(valid_loss_at_epoch)
            valid_acc_per_epoch.append(valid_acc_at_epoch)
            if self.segmentation:
                valid_iou_per_epoch.append(valid_iou_at_epoch)

            # Decide criteria for saving model
            save_model = False
            if self.segmentation:
                if valid_iou_at_epoch > best_iou_score:
                    best_iou_score = valid_loss_at_epoch
                    save_model = True
            else:
                if valid_acc_at_epoch > best_accuracy:
                    best_accuracy = valid_acc_at_epoch
                    save_model = True

            # Save best model
            if save_model:
                early_stop_count = 0
                torch.save(self.model.state_dict(), self.save_path)
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                early_stop_count += 1
                if early_stop_count > early_stop_tolerance:
                    print("Early Stopping...")
                    break
        
        # Save the best weights and return them with all the training data
        self.model.load_state_dict(best_model_weights)

        return self.model, best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch
    
    def train(self, epoch):

        dataset_size = len(self.train_loader.dataset)

        self.model.train(True) # Turn train() back on in case it was turned off
        param_copy = None
        if self.classification:
            running_loss = 0.0
            running_corrects = 0.0
        elif self.segmentation:
            losses = []
            accuracy = []
            mean_iou_scores = []

        for iter, (inputs, labels) in enumerate(self.train_loader):

            inputs =  inputs.to(self.device)
            labels =  labels.to(self.device)
            
            if self.config.augment:
                b, ncrop, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                b, ncrop, h, w = labels.size()
                labels = labels.view(-1, h, w)
            
            self.optimizer.zero_grad()
            with torch.enable_grad(): # torch.set_grad_enabled(True)

                if self.config.debug and epoch >= self.config.debug: 
                    torch.autograd.set_detect_anomaly(True)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                if self.config.clip:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

                if self.config.debug: torch.autograd.set_detect_anomaly(False)

            self.model.train(False)
            if self.config.debug and epoch >= self.config.debug: 
                torch.autograd.set_detect_anomaly(True)
            
            if self.classification:
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            elif self.segmentation:
                losses.append(loss.item())
                acc = util.pixel_acc(preds, labels)
                accuracy.append(acc)
                iou_score = util.iou(preds, labels)
                mean_iou_scores.append(iou_score)

            # if iter % 100 == 0:
            #     print("Iteration[{} / {}] | Loss: {}".format(iter, int(dataset_size/self.train_loader.batch_size), loss.item()))

        if self.classification:
            loss_at_epoch = running_loss / dataset_size
            acc_at_epoch = (running_corrects.double() / dataset_size).item()
            iou_at_epoch = None
        elif self.segmentation:
            loss_at_epoch = np.mean(losses)
            acc_at_epoch = np.mean(accuracy)
            iou_at_epoch = np.mean(mean_iou_scores)

        return loss_at_epoch, acc_at_epoch, iou_at_epoch

    def val(self, epoch):

        dataset_size = len(self.val_loader.dataset)

        self.model.eval() # Put in eval mode (disables batchnorm/dropout) !
        
        if self.classification:
            running_loss = 0.0
            running_corrects = 0.0
        elif self.segmentation:
            losses = []
            accuracy = []
            mean_iou_scores = []

        for iter, (inputs, labels) in enumerate(self.val_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = self.criterion(outputs, labels)

            if self.classification:
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            elif self.segmentation:
                losses.append(loss.item())
                acc = util.pixel_acc(preds, labels)
                accuracy.append(acc)
                iou_score = util.iou(preds, labels)
                mean_iou_scores.append(iou_score)

        if self.classification:
            loss_at_epoch = running_loss / dataset_size
            acc_at_epoch = (running_corrects.double() / dataset_size).item()
            iou_at_epoch = None
        elif self.segmentation:
            loss_at_epoch = np.mean(losses)
            acc_at_epoch = np.mean(accuracy)
            iou_at_epoch = np.mean(mean_iou_scores)

        return loss_at_epoch, acc_at_epoch, iou_at_epoch
    
    def test(self, model=None, data_loader=None):

        if model is None:
            model = self.model
        if data_loader is None:
            test_loader = self.test_loader

        dataset_size = len(test_loader.dataset)

        model.eval()  # Put in eval mode (disables batchnorm/dropout) !

        if self.classification:
            running_loss = 0.0
            running_corrects = 0.0
        elif self.segmentation:
            losses = []
            accuracy = []
            mean_iou_scores = []

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = self.criterion(outputs, labels)

            if self.classification:
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            elif self.segmentation:
                losses.append(loss.item())
                acc = util.pixel_acc(preds, labels)
                accuracy.append(acc)
                iou_score = util.iou(preds, labels)
                mean_iou_scores.append(iou_score)

        if self.classification:
            loss_at_epoch = running_loss / dataset_size
            acc_at_epoch = (running_corrects.double() / dataset_size).item()
            iou_at_epoch = None
        elif self.segmentation:
            loss_at_epoch = np.mean(losses)
            acc_at_epoch = np.mean(accuracy)
            iou_at_epoch = np.mean(mean_iou_scores)

        return loss_at_epoch, acc_at_epoch, iou_at_epoch
    
