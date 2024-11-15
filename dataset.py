from os.path import join
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
import tifffile as tif

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class VGG_SAMPLE_Dataset(Dataset):
    def __init__(self, transform = False, train = True):
        super(VGG_SAMPLE_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()

        path = './Data/SAR Data/SAMPLE/results/231120~/5histogram_matching_gaussian_noise_0.03'
        self.train = train

        if train:
            self.dir_t = 'refine'
        else:
            self.dir_t = 'real'

        self.path2folder = join(path, self.dir_t)

        self.getitem = self._vgg_getitem

        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        
        self.data_name = []
        self.data_label = []
        for label in self.label:
            path2data = join(self.path2folder, label)
            data_name = os.listdir(path2data)
            self.data_name.extend(data_name)

            self.data_label.extend([label] * len(data_name))
    
    def _vgg_getitem(self, index):

        # Label
        label_str = self.data_label[index]
        label = torch.zeros(len(self.label))
        label[self.label.index(label_str)] = 1

        # Data Path
        img_path = join(self.path2folder, label_str, self.data_name[index])

        # Data Load
        img = self.transform(plt.imread(img_path))

        return img, label
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data_name)

if __name__ == '__main__':
    test = VGG_SAMPLE_Dataset(train=False)
    print('a')