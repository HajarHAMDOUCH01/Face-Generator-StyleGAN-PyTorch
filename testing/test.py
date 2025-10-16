import torch
from torch import Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")
from model.style_gan import StyleGAN
from training_config import training_config
import matplotlib.pyplot as plt
import os


def generate_samples(config, model_weights_path=""):
    model = torch.load(model_weights_path, map_location=device)
    img_size = config["image_size"]
    z_dim = config["z_dim"]
    w_dim = config["w_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    generator = StyleGAN(
        z_dim=z_dim,
        w_dim=w_dim,
        img_size=img_size,
        img_channels=3,
        mapping_layers=config["mapping_layers"],
        style_mixing_prob=config["style_mixing_prob"]
    ).to(device)
    generator.load_state_dict(model["generator_state_dict"])
    generator.eval()
    with torch.no_grad():
        z = torch.randn(8, z_dim, device=device)
        samples = generator.generate(z, truncation_psi=1.0)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'Generated Samples')
        save_path = os.path.join("/content/", f'samples_test.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_samples(training_config, model_weights_path="/content/models--hajar001--StyleGAN-Caleba/snapshots/b0ea56a0375134ce5071765dc22b580f621f0f8b/stylegan_checkpoint_epoch_14.pth")
