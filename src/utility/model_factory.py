import __init__

import sys
import os
from collections import Counter

import torch
import torch.nn as nn
import torchvision

import src.utility.util as util
import src.arch as arch

DEFAULT_SIZE_224 = ['import_vit']

def get_weight_initializer():
    def init_weights(module):
        # children_size = len([None for _ in module.children()]) # Its a non-container module if this is 0; but we don't need this
        # module_name = module.__class__.__name__ # For if you want to check which module this is 
        # list(module.parameters()) # This will list out the associated parameters. For non-containers, this is usually between 0-2 (weights and bias)
        invalid_layers = ["BatchNorm2d", "LayerNorm"]
        if module.__class__.__name__ in invalid_layers: return
        try:
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                torch.nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                torch.nn.init.normal_(module.bias.data) # xavier not applicable for biases
        except Exception as E:
            print("Invalid Layer (Please Register It): ", module.__class__.__name__)
            raise E
    return init_weights

def build_model(architecture, classes, image_size, pretrained=False, augment=False):

    class_count = classes if isinstance(classes, int) else len(classes)

    attr = util.Attributes(
        augment = augment
    )

    model_base_transform = None

    if architecture == 'unet':
        model = arch.unet.UNet(n_class=class_count)

    elif architecture == 'transfer':
        model = arch.transfer_fcn.Resnet_FCN(n_class=class_count)

    elif architecture == 'custom1':
        model = arch.customfcn1.Custom_FCN1(n_class=class_count)

    elif architecture == 'custom2':
        model = arch.customfcn2.Custom_FCN2(n_class=class_count)

    elif architecture == 'fcn':
        model = arch.basic_fcn.FCN(n_class=class_count)
    
    elif architecture == 'resnet':
        weights = None if not pretrained else torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights) # /Users/armanommid/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, class_count)
        )

        model_base_transform = weights.transforms

    elif architecture == 'import_vit':
        weights = None if not pretrained else torchvision.models.ViT_B_16_Weights.DEFAULT # IMAGENET1K_SWAG_E2E_V1 # SWAG weights
        model = torchvision.models.vit_b_16(weights=weights)
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
        model.heads = nn.Sequential(
            nn.LayerNorm(normalized_shape=768),
            nn.Linear(in_features=768, out_features=class_count)
        )
        model_base_transform = None if weights is None else weights.transforms

    elif architecture == 'vit' or architecture == 'fvit':
        patching = {
            224 : 16,
            32 : 8,
            28 : 7 
        }

        image_size = image_size
        patch_size = patching[image_size]
        num_layers = 1
        num_heads = 8
        hidden_dim = 32
        expansion = 4
        model = arch.vit.VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, \
                                           hidden_dim=hidden_dim, mlp_dim=(hidden_dim * expansion), num_classes=class_count, fourier=(architecture =='fvit')) #, norm_layer=nn.BatchNorm2d)
        
    else:
        raise NotImplementedError("Model Architecture Not Found")

    attr(model)

    init_weights = get_weight_initializer()
    model.apply(init_weights)

    return model, model_base_transform
