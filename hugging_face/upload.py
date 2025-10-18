"""
Standalone script to upload a trained StyleGAN checkpoint to Hugging Face
No training code needed - just load checkpoint and upload!
"""

import torch
from model import StyleGAN  # Your model.py with the mixin
import os

# ============================================
# CONFIGURATION
# ============================================

# Path to your checkpoint file
CHECKPOINT_PATH = ""

# Hugging Face repository
HF_REPO_ID = ""  

# Model configuration (architecture only)
MODEL_CONFIG = {
    "z_dim": 512,
    "w_dim": 512,
    "img_size": 128,
    "img_channels": 3,
    "mapping_layers": 8,
    "style_mixing_prob": 0.9,
}

# Training info (optional, for model card)
TRAINING_INFO = {
    "dataset": "FFHQ",
    "dataset_size": 70000,
    "trained_epochs": 100,
    "batch_size": 40,
    "learning_rate_g": 0.00025,
    "learning_rate_d": 0.00020,
    "r1_gamma": 5.0,
    "plr_weight": 2.0,
    "style_mixing_prob": 0.8,
}

# Upload settings
PRIVATE_REPO = False  # Set to True for private repository
COMMIT_MESSAGE = "Upload trained StyleGAN model"



print("="*60)
print("LOADING CHECKPOINT")
print("="*60)

print(f"Loading from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

print(f"  Checkpoint loaded")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Keys: {list(checkpoint.keys())}")



print("\n" + "="*60)
print("CREATING GENERATOR")
print("="*60)

# Create generator with your architecture
generator = StyleGAN(
    z_dim=MODEL_CONFIG["z_dim"],
    w_dim=MODEL_CONFIG["w_dim"],
    img_size=MODEL_CONFIG["img_size"],
    img_channels=MODEL_CONFIG["img_channels"],
    mapping_layers=MODEL_CONFIG["mapping_layers"],
    style_mixing_prob=MODEL_CONFIG["style_mixing_prob"],
)

# Load generator weights from checkpoint
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

print(f"✓ Generator created and weights loaded")
print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")



print("\n" + "="*60)
print("TESTING GENERATION")
print("="*60)

try:
    with torch.no_grad():
        z = torch.randn(1, MODEL_CONFIG["z_dim"])
        test_image = generator.generate(z, truncation_psi=0.7)
    print(f"Generation test passed")
    print(f"  Output shape: {test_image.shape}")
except Exception as e:
    print(f"Generation test failed: {e}")
    print("  Proceeding anyway, but model might not work correctly...")


print("\n" + "="*60)
print("UPLOADING TO HUGGING FACE")
print("="*60)

try:
    # Make sure you're logged in first:
    # Run in terminal: huggingface-cli login
    
    print(f"Uploading to: {HF_REPO_ID}")
    print(f"Private: {PRIVATE_REPO}")
    
    generator.push_to_hub(
        repo_id=HF_REPO_ID,
        config=MODEL_CONFIG,  
        commit_message=COMMIT_MESSAGE,
        private=PRIVATE_REPO,
        model_card_kwargs=TRAINING_INFO,  
    )
    
    print("\n" + "="*60)
    print("UPLOAD SUCCESSFUL!")
    print("="*60)
    print(f"\nYour model is now available at:")
    print(f"   https://huggingface.co/{HF_REPO_ID}")
    print(f"\nUsers can load it with:")
    print(f'   from huggingface_hub import PyTorchModelHubMixin')
    print(f'   model = PyTorchModelHubMixin.from_pretrained("{HF_REPO_ID}")')
    
except Exception as e:
    print("\n" + "="*50)
    print("UPLOAD FAILED")
    print("="*60)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're logged in: huggingface-cli login")
    print("2. Check your internet connection")
    print("3. Verify the repository ID is correct")
    print("4. Ensure you have write access to the repository")


print("\n" + "="*60)
print("SAVING LOCAL BACKUP")
print("="*60)

local_save_dir = "./hf_generator_backup"
os.makedirs(local_save_dir, exist_ok=True)

generator.save_pretrained(
    local_save_dir,
    config=MODEL_CONFIG,
)

print(f"Local backup saved to: {local_save_dir}")
print(f"Files: pytorch_model.bin, config.json, README.md")

print("\n" + "="*60)
print("DONE!")
print("="*60)