import numpy as np
import torch
from matplotlib import pyplot as plt
import functorch

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
palette = np.resize(palette, (len(palette)//3, 3))
    
# move to tensor because vmap needs a function with tensor output
palette = torch.tensor(palette)

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

def pixel_acc_mri(pred, target):
#     https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    a = torch.where(target != 0)
    if a[0].shape[0] != 0 or a[1].shape[0] != 0:
        rmin = torch.min(a[1], 0)[0]
        rmax = torch.max(a[1], 0)[0]
        cmin = torch.min(a[2], 0)[0]
        cmax = torch.max(a[2], 0)[0]
#         print(f'bbox: {rmin}-->{rmax}, {cmin}-->{cmax}')

        cropped_target = target[:,rmin:rmax+1,cmin:cmax+1]
        cropped_pred = pred[:,rmin:rmax+1,cmin:cmax+1]
#     else, shape size is 0
    else:
        cropped_target = target
        cropped_pred = pred
    
    n_correct = (cropped_pred==cropped_target).sum().item()
    n_samples = torch.numel(target)
    return n_correct/n_samples

def plot_train_valid(train_data, valid_data, name='Accuracy'):
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
    plt.savefig(name + '.png')

def palette_map(cls):
    return palette[cls]
    
vectorized_palette_map = functorch.vmap(palette_map)
    
# orig should be 224 x 224 x 3 i.e. an image with RGB channels
# true should be the corresponding 2D matrix of true labels in [0,20]
# pred should be the corresponding 2D matrix of predictions in [0,20]
def save_sample(orig, true, pred, filename):
    f, ax = plt.subplots(1,3)
    ax[0].imshow(np.array(orig, dtype=np.uint8))
    ax[0].set_title("image")
    ax[0].axis('off')
    #ax[1].imshow(np.array(vectorized_palette_map(true), dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(np.array(true, dtype=np.uint8), cmap='gray')
    ax[1].set_title("reference")
    ax[1].axis('off')
    ax[2].imshow(np.array(pred, dtype=np.uint8), cmap='gray')
    ax[2].set_title("prediction")
    ax[2].axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
