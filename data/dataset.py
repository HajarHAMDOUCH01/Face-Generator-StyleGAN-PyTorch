from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class FFHQImageDataset(Dataset):
    """
    Simple and efficient dataset for loading images from a directory.
    """
    def __init__(self, root, transform=None, limit=None, extensions=('.jpg', '.jpeg', '.png')):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.root.glob(f'**/*{ext}')))
        
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root} with extensions {extensions}")
        
        print(f"Found {len(self.image_paths)} images in {root}")
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.image_paths)