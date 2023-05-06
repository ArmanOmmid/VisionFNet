import __init__

import sys
import os
import time
import random
from collections import Counter

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

import src.utility.util as util
import src.utility.voc as voc

# For COV Segmentation
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train_val_split(train_val_dataset, val_proportion, targets='targets', slice_targets=False):
    train_targets = np.array(getattr(train_val_dataset, targets)) if not slice_targets else np.array(getattr(train_val_dataset, targets))[:, 1]
    count_range = np.arange(train_targets.shape[0])
    train_indices, val_indices = train_test_split(count_range, test_size=val_proportion, stratify=train_targets)

    train_dataset = torch.utils.data.Subset(train_val_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(train_val_dataset, val_indices)

    # Subset.dataset.indices for indices
    return train_dataset, val_dataset

def prepare_loaders(data_folder_path, dataset_name, transform, batch_size=8, num_workers=1, download=False):
    data_folder_path = data_folder_path.rstrip('/') + '/'

    # Classification 
    if dataset_name == "CIFAR10": # 10 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6)
        class_names = train_val_dataset.classes

    elif dataset_name == "CIFAR100": # 100 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6)
        class_names = train_val_dataset.classes 

    elif dataset_name == "MNIST": # 10 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6)
        class_names = train_val_dataset.classes 

    elif dataset_name == "FashionMNIST": # 10 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6)
        class_names = train_val_dataset.classes 

    elif dataset_name == "StanfordCars": # 196 Classes
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6, targets='_labels')
        class_names = train_val_dataset.classes 

    elif dataset_name == "Food101": # 101 Classes ; HUGE dataset...
        train_val_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/train", download=download, train=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(data_folder_path + "CIFAR10/test", download=download, train=False, transform=transform)
        train_dataset, val_dataset = train_val_split(train_val_dataset, 1/6, targets='_samples', slice_targets=True)
        class_names = train_val_dataset.classes 

    elif dataset_name == "Caltech256": # 256 Classes
        Caltech256 = torchvision.datasets.Caltech256(data_folder_path + "Caltech256", download=download, transform=transform)
        train__val_dataset, test_dataset = train_val_split(Caltech256, test_size=1/6, targets='y')
        # TODO : Ensure you can make subsets on subsets
        train_dataset, test_dataset = train_val_split(train__val_dataset, test_size=1/6) # What is the targets on a subset?
        class_names = Caltech256.categories
    
    # Segmentation 
    elif dataset_name == "VOCSegmentation":
        year = '2007'
        train_dataset = torchvision.datasets.VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='train', transform=transform)
        val_dataset = torchvision.datasets.VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='val', transform=transform)
        test_dataset = torchvision.datasets.VOCSegmentation(data_folder_path + "VOCSegmentation", year=year, download=download, image_set='test', transform=transform)
        class_names = ['person', 
                   'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
                   'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 
                   'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor', 
                   'background']

    # Datset Not Found
    else:
        raise Exception("Dataset Not Prepared")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, class_names

def get_train_transform(augment):

    def train_transform(image, mask):
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
        image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])

        if augment:
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

    return train_transform

def valtest_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])
    
    return image, mask

def sample_transform(image, mask):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

def prepare_dataset(data_root, batch_size, augment):

    year="2007"
    if not os.path.exists(data_root):
        voc.download_voc(data_root, year=year)
    train_transform = get_train_transform(augment)

    train_dataset = voc.VOC(data_root, 'train', transforms=train_transform, year=year)
    val_dataset = voc.VOC(data_root, 'val', transforms=valtest_transform, year=year)
    test_dataset = voc.VOC(data_root, 'test', transforms=valtest_transform, year=year)

    ordered_data = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, ordered_data, 21

def get_class_weights(dataset, n_class):
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
    return class_weights


