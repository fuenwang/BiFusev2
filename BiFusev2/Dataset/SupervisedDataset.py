import os
import sys
import cv2
import numpy as np
from imageio import imread
from tqdm import tqdm 
import io
import zipfile
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from .BaseDataset import BaseDataset


def readImage(path, shape):
    img = np.asarray(imread(path, pilmode='RGB'), np.float32) / 255.0
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)

    return img.transpose(2, 0, 1)

def readDepth(path, shape):
    img = np.load(path)
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)

    return img[None, ...]

class SupervisedDataset(BaseDataset):
    def __init__(self, dataset_path, mode, shape, **kwargs):
        assert os.path.isdir(dataset_path) and mode in ['train', 'val']
        super().__init__(**kwargs)
        self.shape = shape
        with open('%s/%s.txt'%(dataset_path, mode), 'r') as f: lst = [os.path.join(*x.rstrip().split(' ')) for x in f]
        rgb_lst = ['%s/%s/color.jpg'%(dataset_path, x) for x in lst]
        depth_lst = ['%s/%s/depth.npy'%(dataset_path, x) for x in lst]
        self.data = list(zip(rgb_lst, depth_lst))
    
    def __getitem__(self, idx):
        rgb_path, depth_path = self.data[idx]

        rgb = readImage(rgb_path, self.shape)
        depth = readDepth(depth_path, self.shape)

        out = {
            'idx': idx,
            'rgb': rgb,
            'depth': depth
        }

        return out
