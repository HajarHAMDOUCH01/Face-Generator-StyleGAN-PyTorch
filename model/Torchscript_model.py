import torch
import torch.nn.functional as F

import sys

sys.path.append('../')
from style_gan import StyleGAN
from training_config import training_config
checkpoint = torch.load("../stylegan_checkpoint_epoch_10 (2).pth", map_location="cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stylegan = StyleGAN(
    z_dim=training_config["z_dim"],
    w_dim=training_config["w_dim"],
    img_size=training_config["image_size"],
    img_channels=3,
    mapping_layers=training_config["mapping_layers"],
    style_mixing_prob=training_config["style_mixing_prob"]
).to()
stylegan.load_state_dict(checkpoint['generator_state_dict'])
stylegan.eval()

batch_size = 1
z_dim = 512  
dummy_input = torch.randn(batch_size, z_dim)

dummy_z2 = torch.randn(batch_size, z_dim)

# Trace the model
with torch.jit.optimized_execution(should_optimize=True):
    scripted_model = torch.jit.script(
        stylegan, 
        (dummy_input, dummy_z2)
    )
scripted_model.save("scripted_model.pt")
