import numpy as np
from torch.utils.data import Subset

TASKS = {
    'classification' : ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'StanfordCars', 'Food101', 'Caltech256'],
    'segmentation' : ['VOCSegmentation']
}
SIZE = {
    'CIFAR10' : (32,32),
    'CIFAR100' : (32,32),
    'MNIST' : (28,28),
    'FashionMNIST' : (28,28),
    'StanfordCars' : (360, 240),
    'Food101' : (512, 512),
    'Caltech256' : (1024, 1024),
    'VOCSegmentation' : (224, 224)
}

def get_pixel_size(dataset):
    if isinstance(dataset, str):
        return SIZE.get(dataset, None)
    else:
        SIZE.get(get_dataset_name(dataset), None)

def get_base_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset # Recurse until we hit the original dataset object
    return dataset

def get_dataset_name(dataset): # Infer this from the class folder name
    dataset = get_base_dataset(dataset)
    return dataset.__class__.__name__

def get_task_type(dataset):
    if not isinstance(dataset, str):
        dataset = get_dataset_name(dataset)

    task_type = None
    if dataset in TASKS['classification']:
        task_type = 'classification'
    elif dataset in TASKS['segmentation']:
        task_type = 'segmentation'
    return task_type

def get_indices(dataset):
    while not isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset.indices

def get_targets(dataset):

    dataset = get_base_dataset(dataset)

    dataset_name = get_dataset_name(dataset)
    target_attribute = None
    slicing = False

    if dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
        target_attribute = 'targets'
    elif dataset_name in ["StanfordCars"]:
        target_attribute = '_labels'
    elif dataset_name in ['Food101']:
        target_attribute = '_samples'
        slicing = True
    elif dataset_name in ['Caltech256']:
        target_attribute = 'y'
    else:
        raise Exception("Targets Attribute for '{}' has not yet been registered".format(dataset_name))
    
    targets = np.array(getattr(dataset, target_attribute))
    if slicing:
        targets = targets[:, 1]

    return targets