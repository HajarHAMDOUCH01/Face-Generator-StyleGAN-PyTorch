from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image


class MemmapDataset(Dataset):
    def __init__(self, memmap_path, dtype, shape, transform=None):
        super().__init__()
        self.memmap_path = memmap_path
        self.dtype = dtype
        self.shape = shape
        self.transform = transform
        self.memmap = None

    def __getitem__(self, index):
        if self.memmap is None:
            self.memmap = np.memmap(
                self.memmap_path,
                dtype=self.dtype,
                mode='r',
                shape=self.shape
            )
        
        img_array = self.memmap[index]
        
        img_pil = Image.fromarray(img_array)
        
        if self.transform:
            img_pil = self.transform(img_pil)
        
        return img_pil

    def __len__(self):
        return self.shape[0]