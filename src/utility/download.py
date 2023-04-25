
import torchvision

def download_voc(root, year="2007"):
    train_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='train')
    val_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='val')
    test_dataset = torchvision.datasets.VOCSegmentation(root=root, year=year, download=True, image_set='test')
    return train_dataset, val_dataset, test_dataset