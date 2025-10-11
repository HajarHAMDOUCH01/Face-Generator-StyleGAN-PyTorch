import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def preprocess_ffhq_to_numpy(input_root, output_root, target_size=128, limit=None):
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = []
    for img_path in sorted(input_path.rglob("*.png")): 
        images.append(img_path)
        if limit and len(images) >= limit:
            break
    
    print(f"Found {len(images)} images to preprocess")
    
    for img_path in tqdm(images, desc="Converting to numpy"):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        filename = img_path.stem + '.npy'  
        np_path = output_path / filename
        
        np.save(np_path, np.array(img))
    
    print(f"Preprocessing complete! Saved {len(images)} files to {output_root}")




# class FFHQDataset(Dataset):
#     def __init__(self, root, transform=None, limit=None):
#         self.root = root
#         self.transform = transform
#         self.images = []

#         # Walk through all subdirectories (00000, 01000, 02000, ...)
#         # and collect full paths of .png files
#         for dirpath, _, filenames in os.walk(root):
#             # Sort directory names numerically, not lexicographically
#             dir_name = os.path.basename(dirpath)
#             try:
#                 dir_num = int(dir_name)
#             except ValueError:
#                 dir_num = None  

#             # Collect image paths
#             for file in filenames:
#                 if file.lower().endswith(".png"):
#                     full_path = os.path.join(dirpath, file)
#                     self.images.append(full_path)
#                     if limit is not None and len(self.images) >= limit:
#                         break

#         # Sort images by directory numeric order (important for FFHQ)
#         self.images.sort(
#             key=lambda path: int(os.path.basename(os.path.dirname(path)))
#         )

#         print(f"Loaded {len(self.images)} images from {root}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)
#         return image
