
import os
import torchvision
from PIL import Image
from torch.utils import data
import numpy as np

class VOC(data.Dataset):
    def __init__(self, root, mode, transforms, year="2007"):

        '''
        color map
        0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
        12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        '''
        
        #Feel free to convert this palette to a map
        self.palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
        #class 1 and so on......

        self.num_classes = 21
        self.ignore_label = 255
        self.root = root

        if not os.path.exists(root): 
            train, val, test = self.download_voc(root, year=year)
        self.imgs = self.make_dataset(root, mode)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        
        self.mode = mode
        self.transforms = transforms
        self.width = 224
        self.height = 224

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = np.array(Image.open(img_path).convert('RGB').resize((self.width, self.height)))
        mask = np.array(Image.open(mask_path).resize((self.width, self.height)))
        
        img, mask = self.transforms(image=img, mask=mask)

        mask[mask == self.ignore_label] = 0

        return img, mask

    def __len__(self):
        return len(self.imgs)
    
    def download_voc(self, root, year="2007"):
        self.train_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='train')
        self.val_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='val')
        self.test_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='test')
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def make_dataset(self, root, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', mode + '.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
        return items
