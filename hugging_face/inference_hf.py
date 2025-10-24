import torch
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download
import sys, os

# Download and load model
model_file = hf_hub_download(
    repo_id="hajar001/stylegan2-ffhq-128",
    filename="style_gan.py"
)
sys.path.insert(0, os.path.dirname(model_file))
from style_gan import StyleGAN

model = StyleGAN.from_pretrained("hajar001/stylegan2-ffhq-128")

# Generate images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

with torch.no_grad():
    z = torch.randn(1, 512, device=device) # the first agument is for the number of generated faces 
    images = model.generate(z, truncation_psi=0.7)

# Denormalize from [-1, 1] to [0, 1]
images = (images + 1) / 2
images = torch.clamp(images, 0, 1)

# Save as grid (2x2)
save_image(images, "generated_faces.png", nrow=2)
print("✓ Saved: generated_faces.png")