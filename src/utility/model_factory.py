import __init__

import sys
import os
from collections import Counter

import torch
import torch.nn as nn
import torchvision

import src.utility.util as util
import src.arch as arch

def get_weight_initializer():
    def init_weights(module):
        # children_size = len([None for _ in module.children()]) # Its a non-container module if this is 0; but we don't need this
        # module_name = module.__class__.__name__ # For if you want to check which module this is 
        # list(module.parameters()) # This will list out the associated parameters. For non-containers, this is usually between 0-2 (weights and bias)
        invalid_layers = ["LayerNorm", "BatchNorm2d", "BatchNorm1d"]
        if module.__class__.__name__ in invalid_layers: return
        try:
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                torch.nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                torch.nn.init.normal_(module.bias.data) # xavier not applicable for biases
        except Exception as E:
            print("Invalid Layer for Xavier (Please Register It): ", module.__class__.__name__)
            raise E
    return init_weights

def build_model(config, classes):

    class_count = classes if isinstance(classes, int) else len(classes)

    if config.model == 'unet':
        model = arch.unet.UNet(n_class=class_count)

    elif config.model == 'transfer':
        model = arch.transfer_fcn.Resnet_FCN(n_class=class_count)

    elif config.model == 'custom1':
        model = arch.customfcn1.Custom_FCN1(n_class=class_count)

    elif config.model == 'custom2':
        model = arch.customfcn2.Custom_FCN2(n_class=class_count)

    elif config.model == 'fcn':
        model = arch.basic_fcn.FCN(n_class=class_count)
    
    elif config.model == 'resnet':
        weights = None if not config.pretrained else torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights) # /Users/armanommid/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, class_count)
        )

    elif config.model == 'vit_standard':
        weights = None if not config.pretrained else torchvision.models.ViT_B_16_Weights.DEFAULT # IMAGENET1K_SWAG_E2E_V1 # SWAG weights
        model = torchvision.models.vit_b_16(weights=weights)
        if config.pretrained:
            for param in model.parameters():
                param.requires_grad = False
        model.heads = nn.Sequential(
            nn.LayerNorm(normalized_shape=768),
            nn.Linear(in_features=768, out_features=class_count)
        )
    elif config.model == 'vit_import':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = torchvision.models.vision_transformer.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count)
    elif config.model == 'vit':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.vit.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count, fourier=False) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'vit_direct':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.vit_direct.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_gft':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_gft.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_fno':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_fno.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_fsa':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_fsa.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_cross':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_cross.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_spectral':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_spectral.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_spectral_layers=config.num_spectral_layers, \
                                                     num_atn_layers=config.num_atn_layers, num_heads=config.num_heads, hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, \
                                                        num_classes=class_count) # norm_layer=nn.BatchNorm2d)
    elif config.model == 'fvit_token':
        mlp_dim = int(config.hidden_dim * config.expansion)
        model = arch.fvit_token.VisionTransformer(image_size=config.image_size, patch_size=config.patch_size, num_layers=config.num_layers, num_heads=config.num_heads, \
                                           hidden_dim=config.hidden_dim, mlp_dim=mlp_dim, num_classes=class_count, config=config) # norm_layer=nn.BatchNorm2d)
    else:
        raise NotImplementedError("Model Architecture Not Found")

    init_weights = get_weight_initializer()
    model.apply(init_weights)

    return model
