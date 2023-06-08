import __init__

import sys
import os
import time
import gc
import random
from collections import Counter
import argparse
import copy
import yaml
import datetime
import json
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
import torchinfo

import numpy as np
from matplotlib import pyplot as plt

import src.utility.util as util
import src.utility.voc as voc
import src.arch as arch
from src.utility.config import Config

from src.engine.experiment import Experiment
from src.utility.model_factory import build_model
from src.utility.data_factory import prepare_loaders, get_class_weights
from src.utility.data_info import get_task_type, get_targets
from src.utility.transforms import sample_transform
from src.utility.interactive import show_data
from src.utility.voc import VOCSegmentation

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('config', type=str,
                    help="Config name under 'configs/")
parser.add_argument('-D', '--data_path', default=False,
                    help="Path to locate (or download) data from")
parser.add_argument('-S', '--save_path', default=False,
                    help="Path to save model weights to")
parser.add_argument('-L', '--load_path', default=False,
                    help="Path to load model weights from")
parser.add_argument('-E', '--experiment_path', default=False,
                    help="Path to save experiment results")
parser.add_argument('-N', '--experiment_name', default=False,
                    help="Path to save experiment results")
parser.add_argument('--download', action='store_true',
                    help="Download dataset if it doesn't exist")

def main(args):

    print('==== Start Main ====')

    experiment_name = args.experiment_name if args.experiment_name else '_'.join(str(datetime.datetime.now()).split(' ')).split('.')[0]

    # Config Path
    config = args.config
    config_folder = os.path.join(__init__.repository_root, 'configs')
    config_path = os.path.join(config_folder, f'{config}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config Path '{config_path}' does not exist. Choose from: \n{os.listdir(config_folder)}")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = Config(config)

    """ Paths """
    data_path = args.data_path
    download = args.download
    save_path = args.save_path
    load_path = args.load_path
    experiment_path = args.experiment_path

    # Dataset
    task_type = get_task_type(config.dataset)
    if task_type is None:
        raise Exception("Invalid Dataset Name ( Or Unregistered in 'src/utility/data_factory :: get_task_type() )")

    # Data Path
    if data_path:
        if not os.path.exists(data_path):
            if not download:
                raise NotADirectoryError("Dataset Path Does Not Exist: '{}'".format(data_path))
    else:
        data_path = os.path.join(__init__.repository_root, "datasets")

    # Save Path
    if save_path:
        save_path_dir = os.path.basename(save_path)
        if not os.path.exists(save_path_dir):
            raise NotADirectoryError("Save Path Directory Does Not Exist {}".format(save_path_dir))
    else:
        save_path = os.path.join(__init__.repository_root, "weights", "model.pth")

    # Load Path
    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError("Load Path Does Not Exist {}".format(load_path))
    # Plots Path
    if experiment_path:
        if str(__init__.repository_root) in os.path.abspath(experiment_path):
            experiment_path = os.path.join(__init__.repository_root, 'experiments', experiment_name) # Always redirect plots to the designated plot folder if its in the repo
        else:
            experiment_path = os.path.join(experiment_path, experiment_name)
    else:
        experiment_path = os.path.join(__init__.repository_root, 'experiments', experiment_name)

    os.makedirs(experiment_path, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(experiment_path, f"{args.config}.yaml"))

    """ Other Values """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Data """
    data_loaders, data_sizes, class_names = prepare_loaders(config, data_path, download=download)
    train_loader, val_loader, test_loader = list(data_loaders.values())

    print(f"Dataset: {config.dataset}")
    print(data_sizes)

    """ Model """
    model = build_model(config, len(class_names))
    model = model.to(device) # transfer the model to the device
    print("Model Architecture: ", config.model)

    """ Criterion """
    if hasattr(model, 'loss_function'):
        criterion = model.loss_function
    elif config.weighted_loss:
        class_weights = get_class_weights(train_loader, len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    """ Debugging """
    # If Debug, set a hook for modules with an arbitrary debug attribute 
    if config.hooks and len(config.hooks) != 0:
        hook_file_path = os.path.join(experiment_path, 'hook.txt')
        def backward_nan_hook(module, grad_input, grad_output):
            for name, param in module.named_parameters():
                if param.isnan().any():
                    with open(hook_file_path, 'w') as hook_file:
                        hook_file.write(f"Found NaN in parameters")
                        hook_file.write(f"{name}\n{param}")
                        hook_file.write(f"Output\n{grad_output}")
                    raise RuntimeError("NaN Encountered in Backward Pass")
        def forward_nan_hook(module, input, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output
            if attention_module := isinstance(module, nn.MultiheadAttention):
                input = input[0]
                other = outputs[:0]
                outputs = [outputs[0]]
            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    with open(hook_file_path, 'w') as hook_file:
                        hook_file.write(f"Module: {module.__class__.__name__}\n")
                        msg = f"Found NaN in output {i} at indices:\n{nan_mask.nonzero()}\nWhere\n{out[nan_mask.nonzero()[:, 0].unique(sorted=True)]}"
                        hook_file.write(msg)
                        hook_file.write(f"\nInputs\n{input}")
                        hook_file.write(f"\nOutputs\n{output}")
                        if attention_module:
                            hook_file.write(f"\nOther\n{other}")
                    raise RuntimeError(f"NaN Encountered in Forward Pass")
        for module in model.modules():
            condition = True # (isinstance(module, nn.LayerNorm) and hasattr(module, 'debug'))
            if 'backward' in config.hooks:
                module.register_full_backward_hook(backward_nan_hook)
            if 'forward' in config.hooks:
                module.register_forward_hook(forward_nan_hook)

    summary_columns =[ "input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"]
    torchinfo.summary(model=model, input_size=(config.batch_size, 3, config.image_size, config.image_size), col_names=summary_columns)

    """ Optimizer """
    learnable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(learnable_parameters, lr=config.learning_rate, weight_decay=0.001)

    """ Learning Rate Scheduler """
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    else:
        scheduler = None 

    """ Experiment """
    print("Initializing Experiments")
    experiment = Experiment(
        config,
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
    
    print("Training")

    results = experiment.run(config.epochs, config.early_stop_tolerance)

    model, \
    best_iou_score, \
    train_loss_per_epoch, \
    train_iou_per_epoch, \
    train_acc_per_epoch, \
    valid_loss_per_epoch, \
    valid_iou_per_epoch, \
    valid_acc_per_epoch = results
    
    print(f"Best IoU score: {best_iou_score}")
    util.plot_train_valid(train_loss_per_epoch, valid_loss_per_epoch, experiment_path, name='loss')
    util.plot_train_valid(train_acc_per_epoch, valid_acc_per_epoch, experiment_path, name='accuracy')
    if task_type == 'segmentation':
        util.plot_train_valid(train_iou_per_epoch, valid_iou_per_epoch, experiment_path, name='iou')
    
    print('-' * 20)
    test_loss, test_acc, test_iou = experiment.test()
    print(f"Test Loss {test_loss}")
    print(f"Test Accuracy {test_acc}")
    if task_type == 'segmentation':
        print(f"Test IoU {test_iou}")

    results_to_dump = {
        'train_loss' : train_loss_per_epoch,
        'train_accuracy' : train_acc_per_epoch,
        'train_iou' : train_iou_per_epoch,
        'val_loss' : valid_loss_per_epoch,
        'val_accuracy' : valid_acc_per_epoch,
        'val_iou' : valid_iou_per_epoch,
        'test_loss' : test_loss,
        'test_accuracy' : test_acc,
        'test_iou' : test_iou
    }
    if task_type != 'segmentation':
        results_to_dump.pop('train_iou')
        results_to_dump.pop('val_iou')
        results_to_dump.pop('test_iou')

    with open(os.path.join(experiment_path, 'results.json'), 'w') as f:
        json.dump(results_to_dump, f, indent=4)
    
    if task_type == 'segmentation':
        # ------ GET SAMPLE IMAGE FOR REPORT -------
        test_sample_dataset = VOCSegmentation(data_path.rstrip('/') + '/' + "VOCSegmentation", year='2007', download=False, image_set='test', transform=sample_transform)
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

