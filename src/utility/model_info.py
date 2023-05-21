import torch
import torchvision
from torchvision import models

MODEL_TRANSFORM = {
    'resnet' : torchvision.models.ResNet50_Weights.DEFAULT.transforms,
    'vit_standard' : torchvision.models.ViT_B_16_Weights.DEFAULT.transforms
}