import torch
import torchvision
from torchvision import models

MODEL_TRANSFORM = {
    'resnet' : torchvision.models.ResNet50_Weights.DEFAULT.transforms,
    'import_vit' : torchvision.models.ViT_B_16_Weights.DEFAULT.transforms
}