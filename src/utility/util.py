
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
#import functorch

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
palette = np.resize(palette, (len(palette)//3, 3))
    
# move to tensor because vmap needs a function with tensor output
palette = torch.tensor(palette)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()
    

class Attributes(object):

    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __call__(self, object):
        object.__dict__.update(self.__dict__)


def iou(pred, target, n_classes = 21):
    ious = []
    for cls in range(n_classes):
        tar_cls = (target==cls)
        pre_cls = (pred==cls)
        tp = (tar_cls*pre_cls).sum().item()
        fp = ((~tar_cls)*pre_cls).sum().item()
        fn = (tar_cls*(~pre_cls)).sum().item()
        union = tp+fp+fn
        if union != 0:
            ious.append(tp/union)
    return np.mean(ious)

def pixel_acc(pred, target):
    n_correct = (pred==target).sum().item()
    n_samples = torch.numel(target)
    return n_correct/n_samples

def plot_train_valid(train_data, valid_data, plots_path, name='Accuracy'):
    epochs = len(train_data)
    fig, ax = plt.subplots()
    
    ax.plot(train_data, label=f"Training {name}")
    ax.plot(valid_data, label=f"Validation {name}")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
    
    ax.set_title(f"Train/Validation {name} per Epoch")
    ax.set_ylabel(name)
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(1, epochs+1))
    plt.savefig(plots_path + '/' + name + '.png')

def palette_map(cls):
    return palette[cls]
    
vectorized_palette_map = torch.vmap(palette_map)
    
# orig should be 224 x 224 x 3 i.e. an image with RGB channels
# true should be the corresponding 2D matrix of true labels in [0,20]
# pred should be the corresponding 2D matrix of predictions in [0,20]
def save_sample(orig, true, pred):
    f, ax = plt.subplots(1,3)
    ax[0].imshow(np.array(orig, dtype=np.uint8))
    ax[1].imshow(np.array(vectorized_palette_map(true), dtype=np.uint8))
    ax[2].imshow(np.array(vectorized_palette_map(pred), dtype=np.uint8))
    plt.savefig('img_and_colormappedpredictions.png')
    

""" Plotting Experiment Results (ALL)"""

def get_results(experiments_path, styles={}, strict=False):

    if len(styles) == 0: strict = False
    strict_keys = styles.keys()

    results = {
        name : os.path.join(experiments_path, name, 'results.json') 
        for name in os.listdir(experiments_path)
        if os.path.exists(os.path.join(experiments_path, name, 'results.json')) and (not strict or name in strict_keys)
    }

    for name in results.keys():
        with open(results[name], 'r') as json_file:
            results[name] = json.loads(json_file.read())

    train_loss = {}
    val_loss = {}
    test_loss = {}

    train_accuracy = {}
    val_accuracy = {}
    test_accuracy = {}

    for name, result in results.items():

        train_loss[name] = result["train_loss"]
        val_loss[name] = result["val_loss"]
        test_loss[name] = np.round(result["test_loss"], 6)

        train_accuracy[name] = result["train_accuracy"]
        val_accuracy[name] = result["val_accuracy"]
        test_accuracy[name] = np.round(result["test_accuracy"], 6)

    all_results = [train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy]

    for name in styles.keys():
        if isinstance(styles[name][0], int):
            styles[name][0] = f'C{styles[name][0]}'

    def order_styles_first(dict, styles):
        dict_set = list(dict.keys())
        styles_set = list(styles.keys())
        combined_set = styles_set + dict_set
        return {name : dict[name] for name in combined_set if name in dict_set}

    train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = [order_styles_first(result, styles) for result in all_results]

    return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy
    
def plot_results(results_dict, title, y_axis_name, styles={}):
    for name, result in results_dict.items():
        x = list(range(len(result)))
        if name in styles:
            style = styles[name]
            plt.plot(x, result, label=name, color=style[0], linestyle=style[1])
        else:
            plt.plot(x, result, label=name)

    plt.xlabel('Epochs')
    plt.ylabel(y_axis_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show

def print_test_results(results, title, max_first):
    print(title)
    print("="*len(title))
    results = sorted(results.items(), key=lambda x:x[1], reverse=max_first)
    max_len = np.max([len(name) for name, _ in results])
    for name, result in results:
        pad = " "*(max_len - len(name) + 2)
        print(f"{name}{pad}: {result}")

def print_best_val_epoch(val_accuracy, title):
    print(title)
    print("="*len(title))
    max_len = np.max([len(name) for name in val_accuracy.keys()])
    for name, exp in val_accuracy.items():
        pad = " "*(max_len - len(name) + 2)
        print(f"{name}{pad}: {np.argmax(exp)}")