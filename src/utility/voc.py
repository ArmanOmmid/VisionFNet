
import os
import torchvision
from PIL import Image
from torch.utils import data
import numpy as np

class VOCSegmentation(data.Dataset):
    def __init__(self, root, download, image_set, transform, year='2007'):

        assert image_set in ['train', 'val', 'test']

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

        self.classes = ['person', 
                   'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
                   'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 
                   'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor', 
                   'background']

        if download:
            # Surrogate
            torchvision.datasets.VOCSegmentation(root=root, image_set=image_set, download=True, year=year)

        self.imgs = self.make_dataset(root, image_set)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        
        self.image_set = image_set
        self.transform = transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = np.array(Image.open(img_path).convert('RGB').resize((self.width, self.height)))
        mask = np.array(Image.open(mask_path).resize((self.width, self.height)))
        
        img, mask = self.transform(image=img, mask=mask)

        mask[mask == self.ignore_label] = 0

        return img, mask

    def __len__(self):
        return len(self.imgs)
    
    def make_dataset(self, root, image_set):
        items = []
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', image_set + '.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
        return items
