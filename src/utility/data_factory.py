import __init__

import sys
import os
import time
import random
from collections import Counter

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import src.utility.util as util
import src.utility.voc as voc
from src.utility.data_info import get_targets, get_base_dataset, get_dataset_name, get_task_type, get_indices
from src.utility.transforms import ViT_Transform, VOC_Transform
from src.utility.voc import VOCSegmentation

def train_val_split(dataset, split_proportion=1/6, indices=None):
    all_targets = get_targets(dataset)
    if indices is None:
        targets = all_targets
        indices = np.arange(targets.shape[0])
    else:
        targets = all_targets[indices]
    train_indices, val_indices = train_test_split(indices, test_size=split_proportion, stratify=targets)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Subset.dataset.indices for indices
    return train_dataset, val_dataset

def prepare_loaders(data_folder_path, dataset_name, transform_info, batch_size=8, num_workers=1, download=False):
    data_folder_path = data_folder_path.rstrip('/') + '/'

    task_type = get_task_type(dataset_name)
    if task_type == 'classification':
        image_size = 224
        transform = ViT_Transform(image_size)
    elif task_type == 'segmentation':
        augment = False
        transform = None
        # Set Transform Manually Below for VOCSegmentation

    # Classification 
    if dataset_name == 'CIFAR10': # 10 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes

    elif dataset_name == 'CIFAR100': # 100 Classes
        train_val_dataset = torchvision.datasets.CIFAR100(data_folder_path + "CIFAR100/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(data_folder_path + "CIFAR100/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes 

    elif dataset_name == 'MNIST': # 10 Classes
        train_val_dataset = torchvision.datasets.MNIST(data_folder_path + "MNIST/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(data_folder_path + "MNIST/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes 

    elif dataset_name == 'FashionMNIST': # 10 Classes
        train_val_dataset = torchvision.datasets.FashionMNIST(data_folder_path + "FashionMNIST/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(data_folder_path + "FashionMNIST/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes 

    elif dataset_name == 'StanfordCars': # 196 Classes
        train_val_dataset = torchvision.datasets.StanfordCars(data_folder_path + "StanfordCars/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.StanfordCars(data_folder_path + "StanfordCars/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes 

    elif dataset_name == 'Food101': # 101 Classes ; HUGE dataset...
        train_val_dataset = torchvision.datasets.Food101(data_folder_path + "Food101/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(data_folder_path + "Food101/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset)
        class_names = train_val_dataset.classes 

    elif dataset_name == 'Caltech256': # 256 Classes
        Caltech256 = torchvision.datasets.Caltech256(data_folder_path + "Caltech256", download=download, transform=transform)
        train_val_dataset, test_dataset = train_val_split(Caltech256)
        train_dataset, val_dataset = train_val_split(Caltech256, indices=train_val_dataset.indices)
        class_names = Caltech256.categories
    
    # Segmentation 
    elif dataset_name == 'VOCSegmentation':
        year = '2007'
        train_dataset = VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='train', transform=VOC_Transform(augment))
        val_dataset = VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='val', transform=VOC_Transform())
        test_dataset = VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='test', transform=VOC_Transform())
        class_names = train_dataset.classes

    # Datset Not Found
    else:
        raise Exception("Dataset Not Prepared")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names

def get_class_weights(dataset, n_class):
    task_type = get_task_type(dataset)

    if task_type == 'segmentation':
        dataset = get_base_dataset(dataset)
        dataset = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        cum_counts = torch.zeros(n_class)
        for iter, (inputs, labels) in enumerate(dataset):
            labels = torch.squeeze(labels) # 224 x 224
            vals, counts = labels.unique(return_counts = True)
            for v, c in zip(vals, counts):
                cum_counts[v.item()] += c.item()
            #print(f"Cumulative counts at iter {iter}: {cum_counts}")
        totalPixels = torch.sum(cum_counts)
        class_weights = 1 - (cum_counts / totalPixels)
        print(f"Class weights: {class_weights}")

    elif task_type == 'classification':
        indices = get_indices(dataset)
        targets = get_targets(dataset)[indices]
        class_weights = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(targets),
                                        y=targets)
        class_weights = torch.from_numpy(class_weights)
        print(f"Class weights: {class_weights}")
        
    return class_weights
