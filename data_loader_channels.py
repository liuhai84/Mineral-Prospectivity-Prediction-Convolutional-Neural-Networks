from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()

        with open(datatxt) as f:
            txt = f.read()
        imgs = []
        data_info = txt.split('\n')
        for i in data_info:
            info = i.split(' ')
            imgs.append([info[0], int(info[1])])


        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = np.load(fn) 
        if self.transform is not None:
            img = self.transform(img)  
        return img, label
#
    def __len__(self):
        return len(self.imgs)

