#from basic_fcn import *
#import transfer_fcn 
#import customfcn1
#import customfcn2
#import voc
#import util
import torch.nn as nn
import time
import torch
import gc
import os
#import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
import numpy as np
import models.unet as unet
from mri_imgmask import *

import utility.util as util

#MODE = ['lr', 'weight', 'augment', 'unet']
MODE = ['lr', 'weight', 'unet']
"""
None: baseline
'lr': 4a (lr schedule)
'augment': 4b (data augment)
'weight': 4c (weight)
'custom1': 5a-1 (custom1)
'custom2': 5a-2 (custom2)
'transfer': 5b (transfer)
'unet': 5c (unet)
"""

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if 'transfer' in MODE:
        if isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases
    else:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

def getClassWeights(dataset, n_class):
    cum_counts = torch.zeros(n_class)
    for iter, (inputs, labels) in enumerate(train_loader_no_shuffle):
        # inputs: 1x3x256x256, labels: 1x256x256 => 256x256
        labels = torch.squeeze(labels)
        vals, counts = labels.unique(return_counts = True)
        for v, c in zip(vals, counts):
            cum_counts[v.item()] += c.item()
            
        #print(f"Cumulative counts at iter {iter}: {cum_counts}")
            
    totalPixels = torch.sum(cum_counts)
    classWeights = 1 - (cum_counts / totalPixels)
    print(f"Class weights: {classWeights}")
    return classWeights

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])
    
    # mask value from 255 to 1
    mask[mask==255] = 1

    if 'augment' in MODE:
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

def valtest_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])

    # mask value from 255 to 1
    mask[mask==255] = 1
    
    return image, mask

def sample_transform(image, mask):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    
    return image, mask

"""
train_dataset = voc.VOC('train', transforms=train_transform)
val_dataset = voc.VOC('val', transforms=valtest_transform)
test_dataset = voc.VOC('test', transforms=valtest_transform)

if 'augment' in MODE:
    train_loader = DataLoader(dataset=train_dataset, batch_size= 8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= 8, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= 8, shuffle=False)
else:
    train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)
"""
root = os.path.join('archive','lgg-mri-segmentation','kaggle_3m')
make_trainvaltestCSV(root)

train_loader = get_dataloader(root, 'train', transforms=train_transform, batch_size=8, shuffle=True)
valid_loader = get_dataloader(root, 'valid', transforms=valtest_transform, batch_size=8, shuffle=True)
test_loader = get_dataloader(root, 'test', transforms=valtest_transform, batch_size=8, shuffle=True)
train_loader_no_shuffle = get_dataloader(root, 'train', transforms=train_transform, batch_size=1, shuffle=False)

epochs = 20
n_class = 2
learning_rate = 0.01
early_stop_tolerance = 8
model_save_path = 'model.pth'


if 'unet' in MODE:
    model = unet.UNet(n_class=n_class)
""" # delete for project
elif 'transfer' in MODE:
    model = transfer_fcn.Resnet_FCN(n_class=n_class)
elif 'custom1' in MODE:
    model = customfcn1.Custom_FCN1(n_class=n_class)
elif 'custom2' in MODE:
    model = customfcn2.Custom_FCN2(n_class=n_class)
else:
    model = FCN(n_class=n_class)
"""    

model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # determine which device to use (cuda or cpu)

if 'transfer' in MODE:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.Adam(params_to_update, lr = learning_rate, weight_decay=0.001)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

if 'lr' in MODE:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

if 'weight' in MODE:
    classWeights = getClassWeights(train_loader_no_shuffle, n_class).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights) 
else:
    criterion = nn.CrossEntropyLoss() 

model = model.to(device) # transfer the model to the device

def train():
    best_iou_score = 0.0
    best_loss = 100.0
    early_stop_count = 0
    train_loss_per_epoch = []
    train_iou_per_epoch = []
    train_acc_per_epoch = []
    
    valid_loss_per_epoch = []
    valid_iou_per_epoch = []
    valid_acc_per_epoch = []
    
    for epoch in range(epochs):
        ts = time.time()
        losses = []
        mean_iou_scores = []
        accuracy = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# transfer the input to the same device as the model's
            labels =  labels.to(device) # transfer the labels to the same device as the model's
            
            if 'augment' in MODE:
                # due to crop transform
                b, ncrop, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                b, ncrop, h, w = labels.size()
                labels = labels.view(-1, h, w)
            
            outputs = model(inputs) # Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss = criterion(outputs, labels)  # calculate loss

            with torch.no_grad():
                losses.append(loss.item())
                _, pred = torch.max(outputs, dim=1)
                acc = util.pixel_acc(pred, labels)
                accuracy.append(acc)
                iou_score = util.iou(pred, labels)
                mean_iou_scores.append(iou_score)
                
            # backpropagate
            loss.backward()

            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        if 'lr' in MODE:
            print(f'Learning rate at epoch {epoch}: {scheduler.get_lr()[0]:0.9f}')  # changes every epoch
            # lr scheduler
            scheduler.step()           
                    
        with torch.no_grad():
            train_loss_at_epoch = np.mean(losses)
            train_iou_at_epoch = np.mean(mean_iou_scores)
            train_acc_at_epoch = np.mean(accuracy)

            train_loss_per_epoch.append(train_loss_at_epoch)
            train_iou_per_epoch.append(train_iou_at_epoch)
            train_acc_per_epoch.append(train_acc_at_epoch)

            print("Finishing epoch {}, time elapsed {}".format(epoch, time.time() - ts))

            valid_loss_at_epoch, valid_iou_at_epoch, valid_acc_at_epoch = val(epoch)
            valid_loss_per_epoch.append(valid_loss_at_epoch)
            valid_iou_per_epoch.append(valid_iou_at_epoch)
            valid_acc_per_epoch.append(valid_acc_at_epoch)

            if valid_iou_at_epoch > best_iou_score:
                best_iou_score = valid_iou_at_epoch
                # save the best model
            if valid_loss_at_epoch < best_loss:
                print(f"Valid Loss {valid_loss_at_epoch} < Best Loss {best_loss}. (Valid IOU {valid_iou_at_epoch}) Saving Model...")
                best_loss = valid_loss_at_epoch
                early_stop_count = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                early_stop_count += 1
                if early_stop_count > early_stop_tolerance:
                    print("Early Stopping...")
                    break
    model.load_state_dict(torch.load(model_save_path))
            
    return best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch
    

def val(epoch):
    model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        for iter, (input, label) in enumerate(valid_loader):
            input = input.to(device)
            label = label.to(device)
            
            output = model(input)
            loss = criterion(output, label)
            losses.append(loss.item())
            _, pred = torch.max(output, dim=1)
            acc = util.pixel_acc(pred, label)
            accuracy.append(acc)
            iou_score = util.iou(pred, label)
            mean_iou_scores.append(iou_score)
        loss_at_epoch = np.mean(losses)
        iou_at_epoch = np.mean(mean_iou_scores)
        acc_at_epoch = np.mean(accuracy)

    print(f"Valid Loss at epoch: {epoch} is {loss_at_epoch}")
    print(f"Valid IoU at epoch: {epoch} is {iou_at_epoch}")
    print(f"Valid Pixel acc at epoch: {epoch} is {acc_at_epoch}")

    model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return loss_at_epoch, iou_at_epoch, acc_at_epoch


def modelTest():
    model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = criterion(output, label)
            losses.append(loss.item())
            _, pred = torch.max(output, dim=1)
            acc = util.pixel_acc(pred, label)
            accuracy.append(acc)
            iou_score = util.iou(pred, label)
            mean_iou_scores.append(iou_score)

    test_loss = np.mean(losses)
    test_iou = np.mean(mean_iou_scores)
    test_acc = np.mean(accuracy)

    model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return test_loss, test_iou, test_acc


if __name__ == "__main__":

    val(0)  # show the accuracy before training
    
    best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch = train()
    print(f"Best IoU score: {best_iou_score}")
    util.plot_train_valid(train_loss_per_epoch, valid_loss_per_epoch, name='Loss')
    util.plot_train_valid(train_acc_per_epoch, valid_acc_per_epoch, name='Accuracy')
    util.plot_train_valid(train_iou_per_epoch, valid_iou_per_epoch, name='Intersection over Union')
    
    test_loss, test_iou, test_acc = modelTest()
    print(f"Test Loss is {test_loss}")
    print(f"Test IoU is {test_iou}")
    print(f"Test Pixel acc is {test_acc}")
    

    """
    # ------ GET SAMPLE IMAGE FOR REPORT -------
    test_sample_dataset = voc.VOC('test', transforms=sample_transform)
    test_sample_loader = DataLoader(dataset=test_sample_dataset, batch_size=1, shuffle=False)
    model.eval()
    # untransformed original image
    orig_inp, _ = next(iter(test_sample_loader))
    
    # transformed image for input to network
    inp, label = next(iter(test_loader))
    inp = inp.to(device)
    label = label.to(device)
    output = model(inp)
    _, pred = torch.max(output, dim=1)

    util.save_sample(np.array(orig_inp[0].cpu(), dtype=np.uint8), label[0].cpu(), pred[0].cpu())
    model.train()
    # -------------------------------------------
    """
    
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()

