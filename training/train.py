import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc
import os
import numpy as np

import sys 
sys.path.append("/content/VAE_StyleGanDecoder")

from model.vae_model import ConvolutionalVAE
from data.caleba import FacesDataset
from losses import vae_loss_with_perceptual, VGG19
from training_config import training_config


checkpoint_path = None
dataset_path = training_config["dataset_path"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((training_config["image_input_size"], training_config["image_input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def check_for_anomalies(tensor, name="tensor"):
    """Check for NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        return True
    return False


model = ConvolutionalVAE(
    image_channels=3, 
    z_dim=training_config["z_dim"],  
    input_size=training_config["image_input_size"],
    use_style_mixing=True
).to(device)

optimizer = optim.Adam(
    model.parameters(), 
    lr=training_config["lr"], 
    betas=(training_config["adam_beta1"], training_config["adam_beta2"]), 
    eps=training_config["adam_eps"]
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10, 
)

model.train()


def load_model_from_checkpoint(checkpoint_path, model, optimizer):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    if 'beta' in checkpoint:
        beta = checkpoint['beta']
    else:
        beta = training_config["beta_start"]
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Will resume training from epoch {start_epoch}")
    print(f"Current beta: {beta}")

    return model, optimizer, start_epoch, beta


def train_stylegan_vae(model, optimizer, dataset_path, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vgg19_model = VGG19()
    vgg19_model.to(device)
    vgg19_model.eval()

    clear_memory()
    
    z_dim = training_config["z_dim"]
    batch_size = training_config["batch_size"]
    lr = training_config["lr"]
    num_epochs = training_config["num_epochs"]
    beta_start = training_config["beta_start"]
    beta_end = training_config["beta_end"]
    beta_warmup_epochs = training_config["beta_warmup_epochs"]
    free_bits = training_config.get("free_bits", 0.5)
    grad_clip = training_config.get("grad_clip", 0.5)
    
    start_epoch = 0
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kld_losses = []
    epoch_betas = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, beta = load_model_from_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        if checkpoint_path:
            print(f"Checkpoint file {checkpoint_path} not found. Starting fresh training.")
        print("Starting training from scratch")
        beta = beta_start

    transform = get_transforms()
    faces_dataset = FacesDataset(root=dataset_path, transform=transform, limit=10000)
    dataloader = DataLoader(
        faces_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=training_config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if training_config["num_workers"] > 0 else False
    )

    print(f"\nTraining Configuration:")
    print(f"- Z dimension: {z_dim}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {lr}")
    print(f"- Beta schedule: {beta_start} → {beta_end} over {beta_warmup_epochs} epochs")
    print(f"- Free bits: {free_bits}")
    print(f"- Gradient clip: {grad_clip}")
    print(f"- Image range: [-1, 1] (StyleGAN format)")
    print(f"- Total epochs: {num_epochs}\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        num_batches = 0
        
        mu_stats = []
        logvar_stats = []

        if epoch < beta_warmup_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / beta_warmup_epochs)
        else:
            beta = beta_end
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device, non_blocking=True)
            batch_size_actual = real_images.size(0)
            
            optimizer.zero_grad()
            
            recon_imgs, mu, logvar = model(real_images)
            
            if check_for_anomalies(mu, "mu") or check_for_anomalies(logvar, "logvar"):
                print(f"Anomaly detected at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            mu_stats.append(mu.detach().mean().item())
            logvar_stats.append(logvar.detach().mean().item())
            
            use_perceptual = (epoch >= training_config["perceptual_start_epoch"])

            loss, kld_loss, recon_loss = vae_loss_with_perceptual(
                vgg19_model=vgg19_model,
                recon_x=recon_imgs,
                x=real_images,
                mu=mu,
                logvar=logvar,
                beta=beta,
                mse_weight=training_config["mse_weight"],
                percep_weight=training_config["percep_weight"],
                use_percep=use_perceptual,
                free_bits=free_bits
            )
            
            if check_for_anomalies(loss, "loss"):
                print(f"Loss anomaly at epoch {epoch+1}, batch {batch_idx}")
                continue

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()
            
            total_loss += loss.detach().item()
            total_recon += recon_loss.detach().item()
            total_kld += kld_loss.detach().item()
            num_batches += 1
            
            loop.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}", 
                "KLD": f"{kld_loss.item():.4f}",
                "Beta": f"{beta:.4f}",
                "μ": f"{mu.mean().item():.3f}",
                "σ²": f"{logvar.exp().mean().item():.3f}"
            })

            del recon_imgs, mu, logvar, loss, recon_loss, kld_loss

            if batch_idx % 100 == 0:
                clear_memory()
        
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        avg_mu = np.mean(mu_stats)
        avg_logvar = np.mean(logvar_stats)
        
        epoch_losses.append(avg_loss)
        epoch_recon_losses.append(avg_recon)
        epoch_kld_losses.append(avg_kld)
        epoch_betas.append(beta)

        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  KLD Loss: {avg_kld:.4f}")
        print(f"  Beta: {beta:.4f}")
        print(f"  Avg μ: {avg_mu:.4f}")
        print(f"  Avg log(σ²): {avg_logvar:.4f}")
        
        scheduler.step(avg_loss)

        clear_memory()

        if (epoch + 1) % training_config["save_every"] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'beta': beta,
                'losses': {
                    'total': epoch_losses,
                    'recon': epoch_recon_losses,
                    'kld': epoch_kld_losses,
                    'betas': epoch_betas
                }
            }
            checkpoint_file = f'{training_config["save_dir"]}/stylegan_vae_checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")
            
            model_file = f'{training_config["save_dir"]}/stylegan_vae_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_file)
        
        if (epoch + 1) % training_config["sample_every"] == 0:
            generate_samples(model, device, epoch + 1, training_config["save_dir"])
    
    clear_memory()
        
    print("\nTraining complete! Saving final model...")
    final_model_path = f'{training_config["save_dir"]}/stylegan_vae_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    plot_training_curves(epoch_losses, epoch_recon_losses, epoch_kld_losses, epoch_betas)

    return model, epoch_losses


def generate_samples(model, device, epoch, save_dir, num_samples=16):
    """Generate samples from random latent codes"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim).to(device)
        
        samples = model.decode(z)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'Generated Samples - Epoch {epoch}')
        plt.savefig(f'{save_dir}/samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()


def plot_training_curves(epoch_losses, epoch_recon_losses, epoch_kld_losses, epoch_betas):
    """Plot training metrics"""
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(epoch_losses, label='Total Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.plot(epoch_recon_losses, label='Reconstruction Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    plt.plot(epoch_kld_losses, label='KLD Loss', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    plt.plot(epoch_betas, label='Beta Schedule', color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Beta')
    plt.title('Beta Annealing Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{training_config["save_dir"]}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    model, losses = train_stylegan_vae(model, optimizer, dataset_path, checkpoint_path)