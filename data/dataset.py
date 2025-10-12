from pathlib import Path
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class FFHQDatasetNumpy(Dataset):
    def __init__(self, root, transform=None, limit=None):
        self.root = root
        self.transform = transform
        self.images = []

        root_path = Path(root)
        for file_path in sorted(root_path.glob("*.npy")):  
            self.images.append(str(file_path))
            if limit is not None and len(self.images) >= limit:
                break

        print(f"Loaded {len(self.images)} numpy arrays from {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        image = np.load(img_path)  
        
        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        # image = (image / 127.5) - 1.0
        
        if self.transform:
            image = self.transform(image)
        
        return image