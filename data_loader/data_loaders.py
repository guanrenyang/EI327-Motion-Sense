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

class MotionSenseDataset(Dataset):
    def __init__(self, data_dir,num_features, train: bool, time_slice):
        if train:
            self.data = torch.from_numpy(np.load(data_dir+"/preprocessed/train.npy"))[:, :-1].float() # the laset column is gender
        else:
            self.data = torch.from_numpy(np.load(data_dir+"/preprocessed/test.npy"))[:, :-1].float()
        self.time_slice = time_slice
        self.num_features = num_features
    
    def __len__(self):
        return (self.data.shape[0] - self.time_slice + 1) # the whole temporal input data as an input
    def __getitem__(self, idx):
        data = (self.data[idx:idx+self.time_slice, :self.num_features].unsqueeze(0), self.data[idx, self.num_features].long())
        return data
        
class MotionSenseDataLoader(BaseDataLoader):
    """
    Motion Sense data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, time_slice, num_features,  shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MotionSenseDataset(data_dir,num_features, training, time_slice)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
