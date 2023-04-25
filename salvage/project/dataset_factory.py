import csv
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from constants import ROOT_DATA_DIR, MEAN_STD
import torchvision.transforms.functional as TF

def write_csv(df, path, mode):
    csvFN = os.path.join(path, mode+'.csv')
    if(os.path.exists(csvFN) and os.path.isfile(csvFN)):
      #os.remove(csvFN)
      #print(f"{mode+'.csv'} file deleted. recreate")
      print(f"{mode+'.csv'} file already exist. using the existing file")
    else:
      df.to_csv(csvFN)

def make_trainvaltestCSV(path):
    csvFN = os.path.join(path, 'data.csv')
    df = pd.read_csv(csvFN)
    train_df, test_df = train_test_split(df, test_size=0.2)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)
    write_csv(train_df, path, 'train')
    write_csv(valid_df, path, 'valid')
    write_csv(test_df, path, 'test')
    print(f"Total Number of Patients: {df.size}, Train - {train_df.size}, Valid - {valid_df.size}, Test - {test_df.size}")

def train_transform(image, mask, augment=False):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=MEAN_STD[0], std=MEAN_STD[1])
    
    # mask value from 255 to 1
    mask[mask==255] = 1

    if augment:
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

def valtest_transform(image, mask, augment=False):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
    image = TF.normalize(image, mean=MEAN_STD[0], std=MEAN_STD[1])

    # mask value from 255 to 1
    mask[mask==255] = 1
    
    return image, mask

def sample_transform(image, mask, augment=False):
    image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
    mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()
   
    # mask value from 255 to 1
    mask[mask==255] = 1
    return image, mask

class MRI_ImgMask(data.Dataset):
    def __init__(self, path, mode, transforms, augment=False):
        csvFN = os.path.join(path, mode+'.csv')
        self.patients = []
        with open(csvFN) as f:
            csvdata = csv.DictReader(f, delimiter=',')
            for row in csvdata:
                self.patients.append(row['Patient'])
        
        self.imgmask = []
        dir_list = os.listdir(path)
        for dir_name in dir_list:
            dirs = os.path.join(path, dir_name)
            if os.path.isdir(dirs):
                if dir_name[:12] not in self.patients:
                    continue
                for filename in os.listdir(dirs):
                    if filename.split('.')[0].split('_')[-1] != 'mask':
                        img, mask = os.path.join(dirs, filename), os.path.join(dirs, filename.split('.')[0]+'_mask.tif')
                        self.imgmask.append((img,mask))
        
        print(f"Number of Images for {mode} : {len(self.imgmask)}")
        if len(self.imgmask) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
            
        self.transforms = transforms
        self.augment = augment
            
    def __getitem__(self, index):
        img_path, mask_path = self.imgmask[index]
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # perform transform
        img, mask = self.transforms(image=img, mask=mask, augment=self.augment)
        
        return img, mask
    
    def __len__(self):
        return len(self.imgmask)

def get_dataloader(path, mode, transforms, batch_size, shuffle):
    dataset = MRI_ImgMask(path, mode, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
def get_datasets(config_data):
    root = ROOT_DATA_DIR
    make_trainvaltestCSV(root)
    train_loader = get_dataloader(root, 'train', transforms=train_transform, batch_size=config_data['dataset']['batch_size'], shuffle=True)
    valid_loader = get_dataloader(root, 'valid', transforms=valtest_transform, batch_size=config_data['dataset']['batch_size'], shuffle=True)
    test_loader = get_dataloader(root, 'test', transforms=valtest_transform, batch_size=config_data['dataset']['batch_size'], shuffle=True)

    return train_loader, valid_loader, test_loader
