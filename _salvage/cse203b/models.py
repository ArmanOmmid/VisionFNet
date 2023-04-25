
import torch 
from torch import nn
import torchvision
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from torch.utils.data import Subset
from torchinfo import summary
import copy

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

class SViT:
    def __init__(self, vit, tol=1e-3, kernel='rbf', gamma='auto', classifier="ovo"):
        self.vit = copy.deepcopy(vit)
        self.vit.heads[-1] = nn.Identity()
        self.svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, tol=tol, gamma=gamma, decision_function_shape=classifier))

    def load(self, data_loader):
        batch_embeddings = []
        batch_labels = []
        gpu = torch.cuda.is_available()
        with torch.no_grad():
            for data in tqdm(data_loader):
                inputs, labels = data

                if gpu: inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.vit(inputs)
                
                batch_embeddings.append(outputs.cpu().detach().numpy())
                batch_labels.append(labels.cpu().detach().numpy())
        
        embeddings = []
        labels = []
        for input, label in zip(batch_embeddings, batch_labels):
            for x, y in zip(input, label):
                embeddings.append(x)
                labels.append(y)

        return embeddings, labels

    def fit(self, data_loader):
        embeddings, labels = self.load(data_loader)
        return self.svm.fit(embeddings, labels)

    def score(self, data_loader):
        embeddings, labels = self.load(data_loader)
        return self.svm.score(embeddings, labels)
    
def prepare_model(model, class_count=None, embedding_dim=768, BL=False):

    for param in model.parameters(): param.requires_grad = False

    head_layers = []
    if class_count is not None: 
        if BL: head_layers.append(nn.LayerNorm(normalized_shape=embedding_dim))
        head_layers.append(nn.Linear(in_features=embedding_dim, out_features=class_count))
    else:
        head_layers.append(nn.Identity())
    model.heads = nn.Sequential(*head_layers)
    model_summary = summary(model=model, 
        input_size=(1, 3, model.image_size, model.image_size), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    return model, model_summary
