import torch
import torch.nn.functional as F

import sys

sys.path.append('../')
from vae_model import ConvolutionnalVAE
from training_config import training_config

checkpoint = torch.load("vae_checkpoint_epoch_190.pth", map_location="cpu")

model = ConvolutionnalVAE(image_channels=3, z_dim=training_config["z_dim"], input_size=training_config["image_input_size"]).to("cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.rand(1,3,128,128)

traced_model = torch.jit.script(model)
traced_model.save("model_scripted.pt")
