import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os


class DayNightDataset(Dataset):
    def __init__(self, root_dir = os.getcwd() + '/dataset', transform=None):
        if not os.path.exists(root_dir):
            raise OSError("Wrong Path for Dataset")

        self.root_dir = root_dir
        self.transform = transform

         
    def __len__(self):
        day_imgs = len(os.listdir(self.root_dir + 'training/day'))
        night_imgs = len(os.listdir(self.root_dir + 'training/night'))
        return day_imgs + night_imgs

    def __getitem__(self, index):
        pass


