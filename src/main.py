import __init__

import sys
import os
import time
import gc
import random
from collections import Counter
import argparse

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

from src.engine.experiment import Experiment
from src.utility.data_factory import prepare_loaders, prepare_dataset, sample_transform, get_class_weights
from src.utility.model_factory import build_model

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('architecture', type=str,
                    help='Model Architecture')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8,
                    help='Epochs')
parser.add_argument('-l', '--learning_rate', type=int, default=0.01,
                    help='Learning Rate')
parser.add_argument('-w', '--weighted_loss', action='store_true',
                    help='Weighted Loss')
parser.add_argument('-a', '--augment', action='store_true',
                    help='Augmentation of Data')
parser.add_argument('-s', '--scheduler',  action='store_true',
                    help='Learning Rate Scheduler')

parser.add_argument('-D', '--data_path', default=False,
                    help="Path to locate (or download) data from")
parser.add_argument('-N', '--dataset_name', default=False,
                    help="Name of the dataset")
parser.add_argument('-S', '--save_path', default=False,
                    help="Path to save model weights to")
parser.add_argument('-L', '--load_path', default=False,
                    help="Path to load model weights from")
parser.add_argument('--download', action='store_true',
                    help="Download dataset if it doesn't exist")

def main(args):

    """ Hyperparameters """
    architecture = args.architecture
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weighted_loss = args.weighted_loss
    augment = args.augment # False
    scheduler = args.scheduler

    """ Data """
    data_path = args.data_path
    dataset_name = args.dataset_name
    download = args.download

    """ Weights """
    save_path = args.save_path
    load_path = args.load_path

    if data_path:
        if not os.path.exists(data_path):
            raise NotADirectoryError("Dataset Path Does Not Exist {}".format(data_path)) 
    else:
        data_path = os.path.join(__init__.repository_root, "datasets")

    if not dataset_name:
        dataset_name = "VOCSegmentation"

    if save_path:
        save_path_dir = os.path.basename(save_path)
        if not os.path.exists(save_path_dir):
            raise NotADirectoryError("Save Path Directory Does Not Exist {}".format(save_path_dir))
    else:
        save_path = os.path.join(__init__.repository_root, "weights", "model.pth")

    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError("Load Path Does Not Exist {}".format(load_path))

    """ Other Values """
    early_stop_tolerance = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Data """
    train_loader, val_loader, test_loader, class_names = prepare_loaders(data_path, dataset_name, None, batch_size=8, num_workers=1, download=download)
    
    # train_loader, val_loader, test_loader, ordered, classes = prepare_dataset("datasets/VOC", batch_size, augment)


    print(len(train_loader), len(val_loader), len(test_loader), 21)

    print(train_loader.dataset.__dict__.keys())

    raise Exception("STOP")
    
    if weighted_loss:
        class_weights = get_class_weights(ordered, len(classes)).to(device)

    """ Criteron """
    if weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    """ Model """
    model = build_model(architecture, classes, augment)
    model = model.to(device) # transfer the model to the device

    """ Optimizer """
    if model.transfer:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    """ Learning Rate Scheduler """
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    """ Experiment """
    print("Initializing Experiments")
    experiment = Experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        save_path, 
        scheduler=scheduler
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
    test_sample_dataset = voc.VOC(data_path, 'test', transforms=sample_transform)
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

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)

    gc.collect()
    torch.cuda.empty_cache()

