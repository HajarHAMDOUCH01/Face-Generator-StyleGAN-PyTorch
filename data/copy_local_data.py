import shutil
from pathlib import Path
import sys 
import os 
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")
from data.dataset import FFHQDatasetNumpy
from data.preprocess import preprocess_ffhq_fast
from training_config import training_config

def ensure_local_dataset(config):
    """
    Copy preprocessed dataset from Drive to local Colab storage if needed
    """
    drive_path = config["processed_dataset_path"]
    local_path = "/content/ffhq_128_npy_local"  
    
    if os.path.exists(local_path):
        num_files = len(list(Path(local_path).glob("*.npy")))
        print(f"✓ Found {num_files} preprocessed files in local storage")
        return local_path
    
    print(f"📥 Copying preprocessed dataset from Drive to local storage...")
    print(f"   From: {drive_path}")
    print(f"   To: {local_path}")
    
    shutil.copytree(drive_path, local_path)
    
    num_files = len(list(Path(local_path).glob("*.npy")))
    print(f"✓ Copy complete! {num_files} files ready in local storage")
    
    return local_path

if __name__ == "__main__":
    ensure_local_dataset(training_config)

# def train_stylegan(config, checkpoint_path=None):
#     # ... (your existing setup code) ...
    
#     # ADD THIS BEFORE CREATING DATASET:
#     if config["preprocess_data"]:
#         preprocess_ffhq_fast(
#             input_root=config["dataset_path"],
#             output_root=config["processed_dataset_path"],
#             target_size=128,
#             limit=60000,
#             num_workers=8
#         )
    
#     local_dataset_path = ensure_local_dataset(config)
    
#     # Use LOCAL path for dataset
#     dataset = FFHQDatasetNumpy(
#         root=local_dataset_path,  # ← Use local, not Drive!
#         transform=None,
#         limit=config.get("dataset_limit", None)
#     )
    
#     print(f"✓ Dataset ready: {len(dataset)} images from LOCAL storage")
    