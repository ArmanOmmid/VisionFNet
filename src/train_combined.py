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
from src.utility.data_factory import prepare_dataset, sample_transform, getClassWeights
from src.utility.model_factory import init_weights, build_model

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
                    help='Weighted Loss')
parser.add_argument('-s', '--scheduler',  action='store_true',
                    help='Weighted Loss')
parser.add_argument('-D', '--data_path', type=str, default=os.path.join(__init__.repository_root, "datasets", "VOC"),
                    help='Weighted Loss')
parser.add_argument('-S', '--save_path', type=str, default=os.path.join(__init__.repository_root, "weights", "model.pth"),
                    help='Weighted Loss')
parser.add_argument('-L', '--load_path', default=False,
                    help='Weighted Loss')

def main(args):

    """ Arguments """
    architecture = args.architecture
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weighted_loss = args.weighted_loss
    augment = args.augment # False
    scheduler = args.scheduler

    """ Paths """
    data_path = args.data_path
    save_path = args.save_path
    load_path = args.load_path

    """ Other Values """

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    early_stop_tolerance = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, ordered, classes = prepare_dataset(data_path, batch_size, augment)
    
    if class_weights:
        class_weights = getClassWeights(ordered, len(classes)).to(device)
    else:
        class_weights = False

    """ Model """
    model = build_model(args, classes)
    model = model.to(device) # transfer the model to the device

    """ Optimizer """
    if model.transfer:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    """ Learning Rate Scheduler """
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    """ Criteron """
    if weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    """ Experiment """

    print("Initializing Experiments")

    experiment = Experiment(
        model,
        train_loader,
        val_loader,
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


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)

