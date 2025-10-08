import torch
from torch import Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")
from style_gan import StyleGAN
from training_config import training_config

config = {
        "z_dim" : training_config["z_dim"],
        "w_dim" : training_config["w_dim"],
        "img_size" : training_config["img_size"],
        "img_channels" : training_config["img_channels"],
        "mapping_layers" : training_config["mapping_layers"],
        "style_mixing_prob" : training_config["style_mixing_prob"]}


def load_model(model_weights_path=""):
    model = torch.load(model_weights_path, map_location=device)
    style_gan = StyleGAN(**config)
    style_gan.load_state_dict(model)
    generator = style_gan.generator
    generator.load_state_dict(model.synthe)

def generate_face(z: Tensor, ) -> Tensor:
    z = torch.randn((1,512), device=device)
    w = 