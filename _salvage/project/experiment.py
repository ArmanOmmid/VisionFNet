import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
from datetime import datetime

from constants import ROOT_STAT_DIR
from dataset_factory import get_datasets
from file_utils import read_file_in_dir, write_to_file_in_dir, log_to_file_in_dir
from model_factory import get_model

import copy
import util

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STAT_DIR, self.__name)

        # Load Datasets
        self.__train_loader, self.__valid_loader, self.__test_loader = get_datasets(config_data)

        # Experiment Type
        self.__augment = config_data['experiment']['data_augment']
        self.__lr_schedule = config_data['experiment']['lr_schedule']
        self.__class_weight = config_data['experiment']['class_weight']

        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__early_stop_tolerance = config_data['experiment']['early_stop_tolerance']
        self.__current_epoch = 0
        
        self.__train_losses = []
        self.__train_ious = []
        self.__train_accs = []
        self.__valid_losses = []
        self.__valid_ious = []
        self.__valid_accs = []
        
        self.__best_val_loss = 1000
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # determine which device to use (cuda or cpu)

        # Init Model
        model_dict = get_model(config_data, self.__device)
        self.__model = model_dict['model']
        self.__criterion = model_dict['criterion']
        self.__optimizer = model_dict['optimizer']
        
        if self.__lr_schedule:
            self.__scheduler = model_dict['scheduler']

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STAT_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__train_losses = read_file_in_dir(self.__experiment_dir, 'train_losses.txt')
            self.__train_ious = read_file_in_dir(self.__experiment_dir, 'train_ious.txt')
            self.__train_accs = read_file_in_dir(self.__experiment_dir, 'train_accs.txt')
            self.__valid_losses = read_file_in_dir(self.__experiment_dir, 'valid_losses.txt')                
            self.__valid_ious = read_file_in_dir(self.__experiment_dir, 'valid_ious.txt')                
            self.__valid_accs = read_file_in_dir(self.__experiment_dir, 'valid_accs.txt')                
            self.__current_epoch = len(self.__train_losses)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            self.__best_model = copy.deepcopy(self.__model)
            self.__best_model.load_state_dict(state_dict['best'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):

        best_iou_score = 0.0
        early_stop_count = 0

        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch

            train_loss, train_iou, train_acc = self.__train()
            valid_loss, valid_iou, valid_acc = self.__val()

            if valid_iou > best_iou_score:
                best_iou_score = valid_iou
                print(f"best iou: {best_iou_score}")
                torch.save(self.__model.state_dict(), 'bestiou'+self.__name+'.pth')
                # save the best model
            if valid_loss < self.__best_val_loss:
                self.__log(f"Valid Loss {valid_loss} < Best Loss {self.__best_val_loss}. (Valid IOU {valid_iou}) Saving Model...")
                self.__best_val_loss = valid_loss
                self.__best_model = copy.deepcopy(self.__model)
                early_stop_count = 0
                torch.save(self.__model.state_dict(), self.__name+'.pth')
            else:
                early_stop_count += 1
                if early_stop_count > self.__early_stop_tolerance:
                    self.__log("Early Stopping...")
                    break

            self.__record_stats(train_loss, train_iou, train_acc, valid_loss, valid_iou, valid_acc)
            self.__log_epoch_stats(start_time)
            self.__save_model()

        self.__model.load_state_dict(torch.load(self.__name+'.pth'))
        
        return best_iou_score
        
    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):

        self.__model.train()
        
        ts = time.time()
        losses = []
        mean_iou_scores = []
        accuracy = []
        for iter, (inputs, labels) in enumerate(self.__train_loader):
            # reset optimizer gradients
            self.__optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(self.__device)# transfer the input to the same device as the model's
            labels =  labels.to(self.__device) # transfer the labels to the same device as the model's
            
            if self.__augment:
                # due to crop transform
                b, ncrop, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                b, ncrop, h, w = labels.size()
                labels = labels.view(-1, h, w)
            
            outputs = self.__model(inputs) # Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss = self.__criterion(outputs, labels)  # calculate loss

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
            self.__optimizer.step()

            if iter % 10 == 0:
                self.__log("epoch{}, iter{}, loss: {}".format(self.__current_epoch, iter, loss.item()))

        if self.__lr_schedule:
            self.__log(f'Learning rate at epoch {self.__current_epoch}: {self.__scheduler.get_last_lr()[0]:0.9f}')  # changes every epoch
            # lr scheduler
            self.__scheduler.step()           
                    
        with torch.no_grad():
            train_loss_at_epoch = np.mean(losses)
            train_iou_at_epoch = np.mean(mean_iou_scores)
            train_acc_at_epoch = np.mean(accuracy)
            
            self.__log("Finishing epoch {}, time elapsed {}".format(self.__current_epoch, time.time() - ts))
            
        return train_loss_at_epoch, train_iou_at_epoch, train_acc_at_epoch

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
    
        self.__model.eval() # Put in eval mode (disables batchnorm/dropout) !

        losses = []
        mean_iou_scores = []
        accuracy = []
    
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            for iter, (input, label) in enumerate(self.__valid_loader):
                input = input.to(self.__device)
                label = label.to(self.__device)
                
                output = self.__model(input)
                loss = self.__criterion(output, label)
                losses.append(loss.item())
                _, pred = torch.max(output, dim=1)
                acc = util.pixel_acc(pred, label)
                accuracy.append(acc)
                iou_score = util.iou(pred, label)
                mean_iou_scores.append(iou_score)
            loss_at_epoch = np.mean(losses)
            iou_at_epoch = np.mean(mean_iou_scores)
            acc_at_epoch = np.mean(accuracy)
    
        #print(f"Valid Loss at epoch: {self.__current_epoch} is {loss_at_epoch}")
        self.__log(f"Valid Loss at epoch: {self.__current_epoch} is {loss_at_epoch}")
        #print(f"Valid IoU at epoch: {self.__current_epoch} is {iou_at_epoch}")
        self.__log(f"Valid IoU at epoch: {self.__current_epoch} is {iou_at_epoch}")
        #print(f"Valid Pixel acc at epoch: {self.__current_epoch} is {acc_at_epoch}")
        self.__log(f"Valid Pixel acc at epoch: {self.__current_epoch} is {acc_at_epoch}")
    
        self.__model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
        return loss_at_epoch, iou_at_epoch, acc_at_epoch

        
    #  Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        
        self.__model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    
        image_outputs = []
        image_labels = []
        
        losses = []
        mean_iou_scores = []
        accuracy = []
    
        with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
    
            for iter, (input, label) in enumerate(self.__test_loader):
                input = input.to(self.__device)
                label = label.to(self.__device)
    
                output = self.__model(input)
                loss = self.__criterion(output, label)
                losses.append(loss.item())
                _, pred = torch.max(output, dim=1)
                acc = util.pixel_acc(pred, label)
                accuracy.append(acc)
                iou_score = util.iou(pred, label)
                mean_iou_scores.append(iou_score)
                
                image_outputs.extend(output)
                image_labels.extend(label)
    
        test_loss = np.mean(losses)
        test_iou = np.mean(mean_iou_scores)
        test_acc = np.mean(accuracy)

        self.__log(f"Test Loss: {test_loss}")
        self.__log(f"Test IoU: {test_iou}")
        self.__log(f"Test Acc: {test_acc}")
    
        self.__model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
        return test_loss, test_iou, test_acc, image_outputs, image_labels
    
    def myModel(self):
        return self.__model

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        best_dict = self.__best_model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict(), 'best': best_dict}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, train_iou, train_acc, valid_loss, valid_iou, valid_acc):

        self.__train_losses.append(train_loss)
        self.__train_ious.append(train_iou)
        self.__train_accs.append(train_acc)

        self.__valid_losses.append(valid_loss)
        self.__valid_ious.append(valid_iou)
        self.__valid_accs.append(valid_acc)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'train_losses.txt', self.__train_losses)
        write_to_file_in_dir(self.__experiment_dir, 'valid_losses.txt', self.__valid_losses)

        write_to_file_in_dir(self.__experiment_dir, 'train_ious.txt', self.__train_ious)
        write_to_file_in_dir(self.__experiment_dir, 'valid_ious.txt', self.__valid_ious)
        write_to_file_in_dir(self.__experiment_dir, 'train_accs.txt', self.__train_accs)
        write_to_file_in_dir(self.__experiment_dir, 'valid_accs.txt', self.__valid_accs)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__train_losses[self.__current_epoch]
        valid_loss = self.__valid_losses[self.__current_epoch]
        train_iou = self.__train_ious[self.__current_epoch]
        valid_iou = self.__valid_ious[self.__current_epoch]
        train_acc = self.__train_accs[self.__current_epoch]
        valid_acc = self.__valid_accs[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Train IoU: {}, Train Acc: {}, Val Loss: {}, Valid IoU: {}, Valid Acc: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, 
                                         train_loss, train_iou, train_acc, 
                                         valid_loss, valid_iou, valid_acc, 
                                         str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        
        train_comps = [self.__train_losses, self.__train_ious, self.__train_accs]
        valid_comps = [self.__valid_losses, self.__valid_ious, self.__valid_accs]
        names = ['Loss', 'IoU', 'Pixel Accuracy']
        
        for t_comp, v_comp, name in zip(train_comps, valid_comps, names):
            e = len(t_comp)
            x_axis = np.arange(1, e + 1, 1)
            plt.figure()
            plt.plot(x_axis, t_comp, label='Train'+name)
            plt.plot(x_axis, v_comp, label='Valid'+name)
            plt.xlabel("Epochs")
            plt.legend(loc='best')
            plt.title(self.__name + " Stats Plot - " + name)
            plt.savefig(os.path.join(self.__experiment_dir, f"stat_plot_{name}.png"))
            plt.show()
