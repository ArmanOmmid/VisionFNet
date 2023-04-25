
from models.unet import UNet
import torch
import torch.nn as nn
from constants import ROOT_DATA_DIR, CLASS_WEIGHTS
from data.dataset_factory import get_dataloader, train_transform

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.normal_(m.bias.data) #xavier not applicable for biases

def init_weights_transfer(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.normal_(m.bias.data) #xavier not applicable for biases

def getClassWeights(dataset, n_class):
    if CLASS_WEIGHTS == None:
        cum_counts = torch.zeros(n_class)
        for iter, (inputs, labels) in enumerate(dataset):
            # inputs: 1x3x256x256, labels: 1x256x256 => 256x256
            labels = torch.squeeze(labels)
            vals, counts = labels.unique(return_counts = True)
            for v, c in zip(vals, counts):
                cum_counts[v.item()] += c.item()
                
            print(f"Cumulative counts at iter {iter}: {cum_counts}")
                
        totalPixels = torch.sum(cum_counts)
        classWeights = 1 - (cum_counts / totalPixels)
    else:
        classWeights = torch.tensor(CLASS_WEIGHTS)
    print(f"Class weights: {classWeights}")
    return classWeights

def get_model(config_data, device):

    if config_data['model']['model_type'] == 'UNet':
        model = UNet(n_class=config_data['experiment']['n_class'])
    else:
        raise Exception(f"{config_data['model']['model_type']} Not Implemented")
        
    if config_data['model']['transfer']:
        model.apply(init_weights_transfer)
    else:
        model.apply(init_weights)
    
    if config_data['model']['transfer']:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr = config_data['experiment']['learning_rate'], weight_decay=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = config_data['experiment']['learning_rate'])
    
    if config_data['experiment']['lr_schedule']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config_data['experiment']['num_epochs'])
    
    if config_data['experiment']['class_weight']:
        train_loader_no_shuffle = get_dataloader(ROOT_DATA_DIR, 'train', transforms=train_transform, batch_size=1, shuffle=False)
        classWeights = getClassWeights(train_loader_no_shuffle, config_data['experiment']['n_class']).to(device)
        criterion = nn.CrossEntropyLoss(weight=classWeights) 
    else:
        criterion = nn.CrossEntropyLoss() 
    model = model.to(device)
    model_dict = {'model':model, 'criterion':criterion, 'optimizer':optimizer}
    if config_data['experiment']['lr_schedule']:
        model_dict['scheduler'] = scheduler
    return model_dict
