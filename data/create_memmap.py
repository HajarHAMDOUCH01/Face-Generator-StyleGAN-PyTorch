import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import sys 
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")

from training_config import training_config

def create_memmap_dataset_batched(image_dir, output_file, image_size=128, batch_size=100):
    """
    Convert your existing .npy files into one memmap file using batched processing
    """
    image_files = sorted(Path(image_dir).glob('*.npy'))

    n_images = len(image_files)
    print(f"Found {n_images} images")

    # Create memmap file
    memmap_array = np.memmap(
        output_file,
        dtype='uint8',
        mode='w+',
        shape=(n_images, image_size, image_size, 3)
    )

    # Process in batches
    n_batches = (n_images + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_images)

        # Load batch of images
        batch_images = []
        for img_path in image_files[start_idx:end_idx]:
            img = np.load(img_path)
            batch_images.append(img)

        # Stack and write batch to memmap
        batch_array = np.stack(batch_images, axis=0)
        memmap_array[start_idx:end_idx] = batch_array

        # Flush after each batch
        memmap_array.flush()

        # Clear memory
        del batch_images, batch_array

    print(f"\nCreated memmap file: {output_file}")
    print(f"Size: {Path(output_file).stat().st_size / (1024**3):.2f} GB")

    return n_images

if __name__ == "__main__":
    config = training_config
    n_images = create_memmap_dataset_batched(
        config,
        image_dir=config["processed_dataset_path"],
        output_file='',
        image_size=config["image_size"],
        batch_size=100
    )