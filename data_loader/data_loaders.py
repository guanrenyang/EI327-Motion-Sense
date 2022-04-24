import imp
from time import time
from typing_extensions import Self
from catboost import train
from torchvision import datasets, transforms
from yaml import load
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import torch

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MotionSenseDataset(Dataset):
    def __init__(self, data_dir, train: bool, time_slice):
        if train:
            self.data = torch.from_numpy(np.load(data_dir+"/preprocessed/train.npy"))[:, :-1] # the laset column is gender
        else:
            self.data = torch.from_numpy(np.load(data_dir+"/preprocessed/test.npy"))[:, :-1]
        self.time_slice = time_slice
    
    def __len__(self):
        return (self.data.shape[0] - self.time_slice + 1) # the whole temporal input data as an input
    def __getitem__(self, idx):
        data = (self.data[idx:idx+self.time_slice, :-6], self.data[idx:idx+self.time_slice, -6:])
        return data
        
class MotionSenseEncoderDataLoader(BaseDataLoader):
    """
    Motion Sense data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, time_slice,  shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MotionSenseDataset(data_dir, training, time_slice)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
