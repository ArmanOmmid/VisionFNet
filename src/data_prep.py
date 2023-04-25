
import torch
import torchvision
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion, targets='targets', slice_targets=False):
    full_train_targets = np.array(getattr(full_train_dataset, targets)) if not slice_targets else np.array(getattr(full_train_dataset, targets))[:, 1]
    train_dataset_count = full_train_targets.shape[0]
    if train_count is not None:
        train_proportion = min(1, 1 - (train_count / train_dataset_count))
        train_subset_indices, pruned_indices = train_test_split(np.arange(train_dataset_count), test_size=train_proportion, stratify=full_train_targets)
    else: train_subset_indices = np.arange(train_dataset_count); train_count = train_dataset_count
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices)

    full_test_targets = np.array(getattr(full_test_dataset, targets)) if not slice_targets else np.array(getattr(full_test_dataset, targets))[:, 1]
    test_dataset_count = full_test_targets.shape[0]
    if test_proportion is not None: 
        test_proportion = min(1, 1 - ((train_count * test_proportion) / test_dataset_count))
        test_subset_indices, pruned_indices = train_test_split(np.arange(test_dataset_count), test_size=test_proportion, stratify=full_test_targets)
    else: test_subset_indices = np.arange(test_dataset_count)
    test_dataset = torch.utils.data.Subset(full_test_dataset, test_subset_indices)
    
    return train_dataset, test_dataset

def prepare_CIFAR10(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.CIFAR10(data_folder + "CIFAR10/train", download=True, train=True, transform=transform)
    full_test_dataset = torchvision.datasets.CIFAR10(data_folder + "CIFAR10/test", download=True, train=False, transform=transform)
    class_names = full_train_dataset.classes # CIFAR10_Train.class_to_idx provides the mapping {'class_name' : i}
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion)
    return train_dataset, test_dataset, class_names

def prepare_CIFAR100(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.CIFAR100(data_folder + "CIFAR100/train", download=True, train=True, transform=transform)
    full_test_dataset = torchvision.datasets.CIFAR100(data_folder + "CIFAR100/test", download=True, train=False, transform=transform)
    class_names = full_train_dataset.classes 
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion)
    return train_dataset, test_dataset, class_names

def prepare_MNIST(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.MNIST(data_folder + "MNIST/train", download=True, train=True, transform=transform)
    full_test_dataset = torchvision.datasets.MNIST(data_folder + "MNIST/test", download=True, train=False, transform=transform)
    class_names = full_train_dataset.classes 
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion)
    return train_dataset, test_dataset, class_names

def prepare_FashionMNIST(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.FashionMNIST(data_folder + "FashionMNIST/train", download=True, train=True, transform=transform)
    full_test_dataset = torchvision.datasets.FashionMNIST(data_folder + "FashionMNIST/test", download=True, train=False, transform=transform)
    class_names = full_train_dataset.classes 
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion)
    return train_dataset, test_dataset, class_names

def prepare_Food101(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.Food101(data_folder + "Food101/train", download=True, split="train", transform=transform)
    full_test_dataset = torchvision.datasets.Food101(data_folder + "Food101/test", download=True, split="test", transform=transform)
    class_names = full_train_dataset.classes 
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion, targets='_labels')
    return train_dataset, test_dataset, class_names

def prepare_StanfordCars(data_folder, train_count, test_proportion, transform):
    full_train_dataset = torchvision.datasets.StanfordCars(data_folder + "StanfordCars/train", download=True, split="train", transform=transform)
    full_test_dataset = torchvision.datasets.StanfordCars(data_folder + "StanfordCars/test", download=True, split="test", transform=transform)
    class_names = full_train_dataset.classes 
    train_dataset, test_dataset = reusable_splitter(full_train_dataset, full_test_dataset, train_count, test_proportion, targets='_samples', slice_targets=True)
    return train_dataset, test_dataset, class_names

def prepare_Caltech256(data_folder, train_count, test_proportion, transform):
    # Caltech256 Dataset
    Caltech256 = torchvision.datasets.Caltech256(data_folder + "Caltech256", download=True, transform=transform)

    class_names = Caltech256.categories
    targets = np.array(Caltech256.y)
    dataset_count = len(targets)

    all_test = False
    if test_proportion is None: test_proportion = 1/6; all_test = True
    if train_count is None: train_count = dataset_count * (1 - test_proportion)

    train_count = min(train_count, dataset_count * (1 - test_proportion))
    proportion = min(1, 1 - (train_count / dataset_count)*(1+1e-6))
    train_indices, test_indices = train_test_split(np.arange(dataset_count), test_size=proportion, stratify=targets)
    train_dataset = torch.utils.data.Subset(Caltech256, train_indices)

    if not all_test:
        test_count = len(test_indices)
        test_proportion = max(len(class_names), train_count * test_proportion) / test_count
        _, test_indices = train_test_split(np.arange(test_count), test_size=test_proportion, stratify=targets[test_indices])

    test_dataset = torch.utils.data.Subset(Caltech256, test_indices)

    return train_dataset, test_dataset, class_names

def prepare_loaders(data_folder, dataset_name, total_images, test_proportion, transform, batch_size, num_workers):
    data_folder = data_folder.rstrip('/') + '/'
    if dataset_name == "CIFAR10": # 10 Classes
        train_dataset, test_dataset, class_names = prepare_CIFAR10(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "CIFAR100": # 100 Classes
        train_dataset, test_dataset, class_names = prepare_CIFAR100(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "MNIST": # 10 Classes
        train_dataset, test_dataset, class_names = prepare_MNIST(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "FashionMNIST": # 10 Classes
        train_dataset, test_dataset, class_names = prepare_FashionMNIST(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "StanfordCars": # 196 Classes
        train_dataset, test_dataset, class_names = prepare_StanfordCars(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "Food101": # 101 Classes ; HUGE dataset... do last
        train_dataset, test_dataset, class_names = prepare_Food101(data_folder, total_images, test_proportion, transform)
    elif dataset_name == "Caltech256": # 256 Classes
        train_dataset, test_dataset, class_names = prepare_Caltech256(data_folder, total_images, test_proportion, transform)
    else:
        raise Exception("Dataset Not Prepared")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_names