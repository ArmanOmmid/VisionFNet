from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as TF
#import albumentations as A
#import albumentations.pytorch
import util
import numpy as np
from collections import Counter
import random
from matplotlib import pyplot as plt

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

def getClassWeights(dataset, n_class):
    cum_counts = torch.zeros(n_class)
    for iter, (inputs, labels) in enumerate(train_loader_no_shuffle):
        labels = torch.squeeze(labels) # 224 x 224
        vals, counts = labels.unique(return_counts = True)
        for v, c in zip(vals, counts):
            cum_counts[v.item()] += c.item()
            
        #print(f"Cumulative counts at iter {iter}: {cum_counts}")
            
    totalPixels = torch.sum(cum_counts)
    classWeights = 1 - (cum_counts / totalPixels)
    print(f"Class weights: {classWeights}")
    return classWeights

"""
class MyTransform:
    # Apply Rotate and Horizontal Flip

    def __init__(self, angles):
        self.angles = angles
        return

    def __call__(self, x):
        print(x.size())
        res = [x]
        for angle in self.angles:
            res.append(TF.rotate(x, angle))
        res.append(TF.hflip(x))
        return res

mytransform = MyTransform(angles=[-30,-15,15,30])
"""

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])

    #i, j, h, w = standard_transforms.RandomCrop.get_params(image, (128,128))
    #image = TF.crop(image, i, j, h, w)
    #mask = TF.crop(mask, i, j, h, w)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    
    if random.random() > 0.5:
        mask = mask.unsqueeze(0)
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        mask = mask.squeeze(0)
    return image, mask

def valtest_transform(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=mean_std[0], std=mean_std[1])
    
    return image, mask

def sample_transform(image, mask):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    
    return image, mask

"""
valtest_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ])
train_transform = A.Compose([
    A.pytorch.ToTensorV2(),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
    A.GridDistortion(p=0.5),
    A.Normalize(mean=mean_std[0], std=mean_std[1])
    ])

valtest_transform = A.Compose([
    A.pytorch.ToTensorV2(),
    A.Normalize(mean=mean_std[0], std=mean_std[1])
    ])
"""
"""
train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std), 
        standard_transforms.TenCrop(128),
        standard_transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])), 
    ])

train_target_transform = standard_transforms.Compose([
        MaskToTensor(), 
        standard_transforms.TenCrop(128),
        standard_transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
    ])
valtest_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ])

valtest_target_transform = MaskToTensor()
"""

train_dataset = voc.VOC('train', transforms=train_transform)
val_dataset = voc.VOC('val', transforms=valtest_transform)
test_dataset = voc.VOC('test', transforms=valtest_transform)
#train_dataset = voc.VOC('train', transform=train_input_transform, target_transform=train_target_transform)
#val_dataset = voc.VOC('val', transform=valtest_input_transform, target_transform=valtest_target_transform)
#test_dataset = voc.VOC('test', transform=valtest_input_transform, target_transform=valtest_target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

epochs = 20
n_class = 21
learning_rate = 0.01
early_stop_tolerance = 8
model_save_path = 'model.pth'

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # determine which device to use (cuda or cpu)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr = learning_rate) # choose an optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

train_loader_no_shuffle = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
classWeights = getClassWeights(train_loader_no_shuffle, n_class).to(device)
criterion = nn.CrossEntropyLoss(weight=classWeights) # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

fcn_model = fcn_model.to(device) # transfer the model to the device

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
            
            """
            # due to crop transform
            b, ncrop, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            b, ncrop, h, w = labels.size()
            labels = labels.view(-1, h, w)
            """

            outputs = fcn_model(inputs) # Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

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
                torch.save(fcn_model.state_dict(), model_save_path)
            else:
                early_stop_count += 1
                if early_stop_count > early_stop_tolerance:
                    print("Early Stopping...")
                    break
    fcn_model.load_state_dict(torch.load(model_save_path))
            
    return best_iou_score, train_loss_per_epoch, train_iou_per_epoch, train_acc_per_epoch, valid_loss_per_epoch, valid_iou_per_epoch, valid_acc_per_epoch
    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label = label.to(device)
            
            output = fcn_model(input)
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

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return loss_at_epoch, iou_at_epoch, acc_at_epoch


def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device)
            label = label.to(device)

            output = fcn_model(input)
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

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

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
    

    
    # ------ GET SAMPLE IMAGE FOR REPORT -------
    test_sample_dataset = voc.VOC('test', transforms=sample_transform)
    test_sample_loader = DataLoader(dataset=test_sample_dataset, batch_size=1, shuffle=False)
    fcn_model.eval()
    # untransformed original image
    orig_inp, _ = next(iter(test_sample_loader))
    
    # transformed image for input to network
    inp, label = next(iter(test_loader))
    inp = inp.to(device)
    label = label.to(device)
    output = fcn_model(inp)
    _, pred = torch.max(output, dim=1)

    util.save_sample(np.array(orig_inp[0].cpu(), dtype=np.uint8), label[0].cpu(), pred[0].cpu())
    fcn_model.train()
    # -------------------------------------------
    
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()

