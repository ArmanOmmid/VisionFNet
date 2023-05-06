import numpy as np
from torch.utils.data import Subset

TASKS = {
    'classification' : ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'StanfordCars', 'Food101', 'Caltech256'],
    'segmentation' : ['VOCSegmentation']
}

def get_base_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset # Recurse until we hit the original dataset object
    return dataset

def get_dataset_name(dataset): # Infer this from the class folder name
    dataset = get_base_dataset(dataset)
    print(dataset.__class__.__name__)
    return str(dataset.__class__.__name__)

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