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
from torchvision import transforms
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF

import numpy as np
from matplotlib import pyplot as plt

import src.utility.util as util
import src.utility.voc as voc
import src.arch as arch

from src.engine.experiment import Experiment
from src.utility.model_factory import build_model
from src.utility.data_factory import prepare_loaders, get_class_weights
from src.utility.data_info import get_task_type, get_targets
from src.utility.transforms import sample_transform
from src.utility.interactive import show_data
from src.utility.voc import VOCSegmentation

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
parser.add_argument('-n', '--num_workers',  type=int, default=0,
                    help='Number of GPU Workers (Processes)')

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
parser.add_argument('--pretrained', action='store_true',
                    help="Pretrained Weights")

def main(args):

    """ Hyperparameters """
    architecture = args.architecture
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weighted_loss = args.weighted_loss
    augment = args.augment # False
    scheduler = args.scheduler
    num_workers = args.num_workers

    """ Data """
    data_path = args.data_path
    dataset_name = args.dataset_name
    download = args.download

    """ Weights """
    save_path = args.save_path
    load_path = args.load_path
    pretrained = args.pretrained

    """ Data and Path Preperations """

    if not dataset_name:
        dataset_name = "VOCSegmentation"

    task_type = get_task_type(dataset_name)
    if task_type is None:
        raise Exception("Invalid Dataset Name ( Or Unregistered in 'src/utility/data_factory :: get_task_type() )")

    if data_path:
        if not os.path.exists(data_path):
            raise NotADirectoryError("Dataset Path Does Not Exist {}".format(data_path)) 
    else:
        data_path = os.path.join(__init__.repository_root, "datasets")

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

    transform_info = None # TODO : Options for this? Currently automatic

    """ Data """
    train_loader, val_loader, test_loader, class_names = prepare_loaders(data_path, dataset_name, transform_info=transform_info, 
                                                                         batch_size=batch_size, num_workers=num_workers, download=download)
    data_loaders = {
        'train': train_loader,
        'val' : val_loader,
        'test' : test_loader
    }
    dataset_sizes = {mode: len(loader.dataset) for mode, loader in data_loaders.items()}

    if weighted_loss:
        class_weights = get_class_weights(train_loader, len(class_names)).to(device)

    interactive_data_showcase = False
    if interactive_data_showcase:
        show_data(train_loader, class_names)

    """ Criteron """
    if weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    """ Model """
    model, model_base_transform = build_model(architecture, len(class_names), pretrained, augment)
    model = model.to(device) # transfer the model to the device

    """ Optimizer """
    learnable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(learnable_parameters, lr=learning_rate, weight_decay=0.001)

    """ Learning Rate Scheduler """
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # LR = gamma * LR every 7 epochs

    def train_model(model, criterion, optimizer, scheduler, data_loaders, device, num_epochs=25):
        since = time.time()

        dataset_sizes = {mode: len(loader.dataset) for mode, loader in data_loaders.items()}

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    # model = train_model(model, criterion, optimizer, scheduler, data_loaders, device, num_epochs=25)

    """ Experiment """
    print("Initializing Experiments")
    experiment = Experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_path,
        load_path
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
    test_sample_dataset = VOCSegmentation(data_path.rstrip('/') + '/' + "VOCSegmentation", year='2007', image_set='test', transform=sample_transform)
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

