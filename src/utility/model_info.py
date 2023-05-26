import torch
import torchvision
from torchvision import models
from src.utility.transforms import Basic_Compose, VOC_Transform, PIL_RGB

MODEL_TRANSFORM = {
    'resnet' : torchvision.models.ResNet50_Weights.DEFAULT.transforms,
    'vit_standard' : Basic_Compose(224), # torchvision.models.ViT_B_16_Weights.DEFAULT.transforms
}