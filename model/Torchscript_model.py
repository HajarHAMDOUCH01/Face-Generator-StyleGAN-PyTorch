import torch
import torch.nn.functional as F

import sys

sys.path.append('../')
from style_gan import StyleGAN

checkpoint = torch.load("./checkpoint.pth", map_location="cpu")

stylegan = StyleGAN()
stylegan.load_state_dict(checkpoint['model_state_dict'])
stylegan.eval()

batch_size = 1
z_dim = 512  
dummy_input = torch.randn(batch_size, z_dim)

dummy_z2 = torch.randn(batch_size, z_dim)

# Trace the model
traced_model = torch.jit.trace(
    stylegan, 
    (dummy_input, dummy_z2),  
    strict=False
)
traced_model.save("model_traced.pt")
