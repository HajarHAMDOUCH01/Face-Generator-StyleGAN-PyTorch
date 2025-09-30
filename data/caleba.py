import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import sys 
sys.path.append("/content/Convolutiional_VAE")

from training_config import training_config

class FacesDataset(Dataset):
    def __init__(self, root, transform=None, limit=None):
        self.root = root
        self.transform = transform
        
        self.image_files = [
            f for f in os.listdir(root) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if limit is not None:
            self.image_files = self.image_files[:limit]
        
        print(f"Loaded {len(self.image_files)} images from {root}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_stylegan_transforms(image_size=128):
    """
    Get transforms for StyleGAN-based VAE
    IMPORTANT: Normalizes to [-1, 1] range
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
    ])


def denormalize_images(images):
    """
    Convert images from [-1, 1] back to [0, 1] for visualization
    
    Args:
        images: tensor in [-1, 1] range
    Returns:
        tensor in [0, 1] range
    """
    return (images + 1) / 2


# Test the dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test transforms
    transform = get_stylegan_transforms(128)
    
    dataset = FacesDataset(
        root=training_config["dataset_path"],
        transform=transform,
        limit=16
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(dataset):
            img = dataset[i]
            
            img_viz = denormalize_images(img)
            img_viz = torch.clamp(img_viz, 0, 1)
            
            ax.imshow(img_viz.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150)
    print("Dataset test completed!")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"Expected range: [-1, 1]")