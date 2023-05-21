import __init__

import sys

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
import numpy as np

def Tensor_RGB(x): # torch.transforms.Lambda can't be pickled : transforms.Lambda(lambda x: (x.repeat(3, 1, 1) if x.size(0)==1 else x))
    return x.repeat(3, 1, 1) if x.size(0)==1 else x
def PIL_RGB(x): # torch.transforms.Lambda can't be pickled : transforms.Lambda(lambda x: x.convert('RGB')), # Turns to RGB
    return x.convert('RGB')

def Basic_Compose(image_size):
    basic_tranform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        Tensor_RGB,
    ])
    return basic_tranform

def VOC_Transform(augment=False):

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def VOC_Transform(image, mask):
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

    return VOC_Transform

def sample_transform(image, mask):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    return image, mask