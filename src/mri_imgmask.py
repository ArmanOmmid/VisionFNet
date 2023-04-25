import numpy as np
import pandas as pd
import csv
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

def write_csv(df, path, mode):
    csvFN = os.path.join(path, mode+'.csv')
    if(os.path.exists(csvFN) and os.path.isfile(csvFN)):
      os.remove(csvFN)
      print(f"{mode+'.csv'} file deleted. recreate")
    csvFN = os.path.join(path, mode+'.csv')
    df.to_csv(csvFN)

def make_trainvaltestCSV(path):
    csvFN = os.path.join(path, 'data.csv')
    df = pd.read_csv(csvFN)
    train_df, test_df = train_test_split(df, test_size=0.2)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)
    write_csv(train_df, path, 'train')
    write_csv(valid_df, path, 'valid')
    write_csv(test_df, path, 'test')
    print(f"Total Data: {df.size}, Train - {train_df.size}, Valid - {valid_df.size}, Test - {test_df.size}")

def make_trainvaltestCSV_mini(path, num):
    csvFN = os.path.join(path, 'data.csv')
    df = pd.read_csv(csvFN)
    _, df = train_test_split(df, test_size=(num/df.size))
    train_df, test_df = train_test_split(df, test_size=0.2)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)
    write_csv(train_df, path, 'train')
    write_csv(valid_df, path, 'valid')
    write_csv(test_df, path, 'test')
    print(f"MINI TEST Total Patient: {df.size}, Train - {train_df.size}, Valid - {valid_df.size}, Test - {test_df.size}")
    
def get_dataloader(path, mode, transforms, batch_size, shuffle):
    dataset = MRI_ImgMask(path, mode, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class MRI_ImgMask(data.Dataset):
    def __init__(self, path, mode, transforms):
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
        
        print(f"{mode} : {len(self.imgmask)}")
        if len(self.imgmask) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
            
        self.transforms = transforms
            
    def __getitem__(self, index):
        img_path, mask_path = self.imgmask[index]
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # perform transform
        img, mask = self.transforms(image=img, mask=mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.imgmask)