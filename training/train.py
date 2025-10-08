import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import gc
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from huggingface_hub import HfApi

import sys 
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")

# Initialize once
hf_api = HfApi()


from model.style_gan import StyleGAN, Discriminator
from training_config import training_config
from ADA.ada import ADAugment, AugmentationPipeline


torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def path_length_regularization(fake_images, w, mean_path_length, decay=0.01):
    """
    Perceptual Path Length regularization
    Measures how much images change as we interpolate in W space
    """
    batch_size = fake_images.shape[0]
    
    noise = torch.randn_like(fake_images) / np.sqrt(fake_images.shape[2] * fake_images.shape[3])
    
    # Computes gradients of (image * noise) w.r.t. w
    # This is the Jacobian-vector product
    grad_outputs = (fake_images * noise).sum()
    
    gradients = torch.autograd.grad(
        outputs=grad_outputs,
        inputs=w,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Computes path length: L2 norm of gradients
    # Handles both 2D [batch, w_dim] and 3D [batch, layers, w_dim] shapes
    if gradients.dim() == 3:
        # Sum over w_dim, mean over layers, then sqrt
        path_lengths = torch.sqrt(gradients.pow(2).sum(dim=2).mean(dim=1))
    else:
        path_lengths = torch.sqrt(gradients.pow(2).sum(dim=1))
    
    # Exponential moving average of path length
    path_mean = path_lengths.mean()
    
    if mean_path_length == 0:
        mean_path_length = path_mean.detach()
    else:
        mean_path_length = mean_path_length + decay * (path_mean.detach() - mean_path_length)
    
    # Penalty: squared deviation from mean
    path_penalty = ((path_lengths - mean_path_length) ** 2).mean()
    
    return path_penalty, mean_path_length, path_mean.item()

class CelebADataset(Dataset):
    """CelebA dataset loader"""
    def __init__(self, root, transform=None, limit=None):
        self.root = Path(root)
        self.transform = transform
        self.image_files = sorted(list(self.root.glob("*.jpg")))[:limit]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_transforms(img_size):
    """Data preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])


def compute_gradient_penalty(discriminator, real_images):
    """R1 regularization, this excpects realimages with equies_gad true"""
    real_scores = discriminator(real_images)
    
    gradients = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    
    # R1 penalty: ||∇D(x)||²
    r1_penalty = gradients.pow(2).reshape(gradients.size(0), -1).sum(1).mean()
    return r1_penalty


def d_logistic_loss(real_scores, fake_scores):
    """Non-saturating logistic loss for discriminator"""
    real_loss = F.softplus(-real_scores).mean()
    fake_loss = F.softplus(fake_scores).mean()
    return real_loss + fake_loss


def g_nonsaturating_loss(fake_scores):
    """Non-saturating loss for generator"""
    return F.softplus(-fake_scores).mean()


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def generate_samples(generator, device, epoch, save_dir, num_samples=16):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.z_dim, device=device)
        samples = generator.generate(z, truncation_psi=1.0)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'Generated Samples - Epoch {epoch}')
        save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    generator.train()

def load_from_checkpoint(generator, discriminator, g_optimizer, d_optimizer, ada, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    def strip_prefix(state_dict, prefix="_orig_mod."):
        return {
            k.replace(prefix, ""): v 
            for k, v in state_dict.items()
        }
    
    gen_state = checkpoint["generator_state_dict"]
    if any(k.startswith("synthesis._orig_mod") for k in gen_state.keys()):
        gen_state_fixed = {}
        for k, v in gen_state.items():
            if k.startswith("synthesis._orig_mod."):
                new_key = k.replace("synthesis._orig_mod.", "synthesis.")
                gen_state_fixed[new_key] = v
            else:
                gen_state_fixed[k] = v
        gen_state = gen_state_fixed
    
    generator.load_state_dict(gen_state)
    
    disc_state = checkpoint["discriminator_state_dict"]
    if any("_orig_mod" in k for k in disc_state.keys()):
        disc_state = strip_prefix(disc_state)
    
    discriminator.load_state_dict(disc_state)
    
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1

    if 'ada_state' in checkpoint:
        ada.load_state_dict(checkpoint["ada_state"])
        print(f"Loaded ADA state: p={ada.p:.3f}")
    else:
        print("Warning: No ADA state in checkpoint, starting fresh")
    
    print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return generator, discriminator, g_optimizer, d_optimizer, start_epoch, ada

def train_stylegan(config, checkpoint_path=None):
    
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
    
    discriminator = Discriminator(
        img_size=img_size,
        img_channels=3
    ).to(device)
    
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config["g_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"]
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config["d_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"]
    )


    # scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     g_optimizer, 
    #     T_max=config["num_epochs"],
    #     eta_min=config["adam_eps"]  
    # )

    # scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     d_optimizer, 
    #     T_max=config["num_epochs"],
    #     eta_min=config["adam_eps"]
    # )

    
    # Dataset
    transform = get_transforms(img_size)
    dataset = CelebADataset(
        root=config["dataset_path"],
        transform=transform,
        limit=config["dataset_limit"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
        generator=torch.Generator().manual_seed(epoch)
    )
    
    # print(f"\nTraining Configuration:")
    # print(f"- Image size: {img_size}x{img_size}")
    # print(f"- Batch size: {batch_size}")
    # print(f"- Generator LR: {config['g_lr']}")
    # print(f"- Discriminator LR: {config['d_lr']}")
    # print(f"- R1 gamma: {config['r1_gamma']}")
    # print(f"- Dataset size: {len(dataset)}")
    # print(f"- Total epochs: {num_epochs}\n")


    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        generator, discriminator, g_optimizer, d_optimizer, start_epoch, ada = load_from_checkpoint(
            generator, discriminator, g_optimizer, d_optimizer, checkpoint_path
        )
        print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
    else:
        print(f"Starting training from epoch 0")
    
    # generator.synthesis = torch.compile(generator.synthesis)
    # discriminator = torch.compile(discriminator)

    g_losses = []
    d_losses = []
    r1_penalties = []

    mean_path_length = torch.tensor(0.0, device=device)
    plr_losses = []
    
    # PLR hyperparameters
    plr_weight = config.get("plr_weight", 2.0)  
    plr_interval = config.get("plr_interval", 4)  
    plr_decay = config.get("plr_decay", 0.01) 
    warmup_epochs = 5

    ada = ADAugment(
        target_rt=config["target_rt"],
        adjustment_speed_imgs=config["adjustment_speed_imgs"],
        batch_size=config["batch_size"],
        initial_p=config["initial_p"],
        update_interval=config["update_interval"],
        device=device
    )

    for epoch in range(start_epoch, num_epochs):
        # Dynamic R1 gamma during warmup
        if epoch < warmup_epochs:
            current_r1_gamma = config["r1_gamma"] * 2.0  
        else:
            current_r1_gamma = config["r1_gamma"]
        
        n_critic = 1
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_r1 = 0
        epoch_plr = 0
        num_batches = 0
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, real_images in enumerate(loop):
            # ===== Move real images to device =====
            real_images = real_images.to(device, non_blocking=True)
            batch_size_actual = real_images.size(0)
            
            # ==================== Train Discriminator ====================
            d_loss_accum = 0.0
            r1_penalty_accum = 0.0
            
            for _ in range(n_critic):
                discriminator.requires_grad_(True)
                generator.requires_grad_(False)
                
                d_optimizer.zero_grad()
                
                z1 = torch.randn(batch_size_actual, z_dim, device=device)
                z2 = torch.randn(batch_size_actual, z_dim, device=device)
                
                # Generate fake images
                with torch.no_grad():
                    fake_images, _ = generator(z1, z2)
                
                # ===== Apply ADA augmentation =====
                real_images_aug = ada.apply(real_images)
                fake_images_aug = ada.apply(fake_images)
                
                # Forward through discriminator
                real_scores = discriminator(real_images_aug)
                fake_scores = discriminator(fake_images_aug)
                
                # Base discriminator loss
                d_loss = d_logistic_loss(real_scores, fake_scores)
                
                # ===== R1 Regularization =====
                r1_penalty = torch.tensor(0.0, device=device)
                if batch_idx % config["r1_interval"] == 0:
                    # Option A: Reuse the same augmentation (efficient)
                    # We need to re-augment with requires_grad=True because
                    # the first augmentation was applied to a tensor without requires_grad
                    real_images.requires_grad_(True)  # Enable grad on source
                    real_images_aug_r1 = ada.apply(real_images)
                    
                    r1_penalty = compute_gradient_penalty(discriminator, real_images_aug_r1)
                    
                    # Clear requires_grad after R1 computation
                    real_images.requires_grad_(False)
                    
                    # Lazy R1: scale by interval
                    d_loss_with_r1 = d_loss + (current_r1_gamma / 2) * r1_penalty * config["r1_interval"]
                    r1_penalty_accum += r1_penalty.item()
                else:
                    d_loss_with_r1 = d_loss
                
                # Backward and optimize
                d_loss_with_r1.backward()
                d_optimizer.step()
                
                # Accumulate losses (use the actual optimized loss)
                d_loss_accum += d_loss.item()
            
            # ===== ADA Update (once per batch, outside n_critic loop) =====
            current_p = ada.update(real_scores)
            
            # Accumulate epoch statistics
            epoch_d_loss += d_loss_accum / n_critic
            if r1_penalty_accum > 0:
                epoch_r1 += r1_penalty_accum / n_critic
            
            # ===== Diagnostics (after everything is computed) =====
            if batch_idx == 0 and epoch < 3:
                print(f"\n=== ADA Diagnostic (Epoch {epoch}, Batch {batch_idx}) ===")
                print(f"ADA p: {ada.p:.4f}")
                print(f"ADA buffer size: {len(ada.logits_buffer)}")
                print(f"Real images range: [{real_images.min():.2f}, {real_images.max():.2f}]")
                print(f"Real aug range: [{real_images_aug.min():.2f}, {real_images_aug.max():.2f}]")
                print(f"Fake aug range: [{fake_images_aug.min():.2f}, {fake_images_aug.max():.2f}]")
                print(f"Real logits (first 4): {real_scores[:4].detach().cpu().numpy()}")
                print(f"Fake logits (first 4): {fake_scores[:4].detach().cpu().numpy()}")
                print(f"r_t (if available): {ada.rt_history[-1] if ada.rt_history else 'N/A'}")
                print(f"Real images on device: {real_images.device}")
                print("=" * 60)
        
            
            # ==================== Train Generator ====================
            if batch_idx == 0 and epoch == 0:
                print("\n=== Generator Forward Pass Test ===")
                test_z1 = torch.randn(2, z_dim, device=device)
                test_z2 = torch.randn(2, z_dim, device=device)
                
                test_fake, test_w = generator(test_z1, test_z2, return_w=True)
                print(f"fake_images shape: {test_fake.shape}")
                print(f"w shape: {test_w.shape if test_w is not None else 'None'}")
                print(f"w requires_grad: {test_w.requires_grad if test_w is not None else 'N/A'}")
                print(f"fake_images requires_grad: {test_fake.requires_grad}")
                
                # Test augmentation
                test_aug = ada.apply(test_fake)
                print(f"Augmented range: [{test_aug.min():.2f}, {test_aug.max():.2f}]")
                
                # Test gradient flow
                test_scores = discriminator(test_aug)
                test_loss = test_scores.mean()
                test_loss.backward()
                
                has_grad = any(p.grad is not None for p in generator.parameters())
                print(f"Generator received gradients: {has_grad}")
                print("=" * 50 + "\n")
                
                # Clear test gradients
                generator.zero_grad()
            discriminator.requires_grad_(False)
            generator.requires_grad_(True)

            g_optimizer.zero_grad()

            z1 = torch.randn(batch_size_actual, z_dim, device=device)
            z2 = torch.randn(batch_size_actual, z_dim, device=device)

            # Forward pass through generator
            fake_images, w = generator(z1, z2, return_w=True)

            # Handle case where generator didn't return w (shouldn't happen with return_w=True, but defensive)
            if w is None:
                # Manually compute w
                w = generator.mapping(generator.mapping.pixel_norm(z1))
                fake_images = generator.synthesis(w)

            # Apply ADA augmentation
            fake_images_aug = ada.apply(fake_images)

            # Get discriminator scores on augmented images
            fake_scores = discriminator(fake_images_aug)

            # Non-saturating loss
            g_loss = g_nonsaturating_loss(fake_scores)

            # Path Length Regularization (computed on NON-augmented images)
            plr_loss = torch.tensor(0.0, device=device)
            plr_value = 0.0

            if batch_idx % plr_interval == 0:
                # PLR measures smoothness in generator's native output space
                # We use fake_images (non-augmented) because:
                # 1. PLR is about generator geometry, not discriminator perception
                # 2. Augmentations add noise that would corrupt the path length measurement
                plr_loss, mean_path_length, plr_value = path_length_regularization(
                    fake_images, w, mean_path_length, decay=plr_decay
                )
                g_loss_total = g_loss + plr_weight * plr_loss
                epoch_plr += plr_value
            else:
                g_loss_total = g_loss

            g_loss_total.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()

            num_batches += 1

            loop.set_postfix({
                "G_loss": f"{g_loss.item():.4f}",
                "D_loss": f"{d_loss.item():.4f}",
                "R1": f"{r1_penalty.item():.4f}" if r1_penalty.item() > 0 else "0.0000",
                "PLR": f"{plr_loss.item():.4f}" if plr_loss.item() > 0 else "0.0000",
                "ADA_p": f"{ada.p:.3f}",  # Use ada.p property, not current_p
                "ADA_rt": f"{ada.rt_history[-1]:.3f}" if ada.rt_history else "N/A",
            })
        # ==================== End of Epoch ====================
        # scheduler_g.step()
        # scheduler_d.step()
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_r1 = epoch_r1 / max(1, num_batches // config["r1_interval"])
        avg_plr = epoch_plr / max(1, num_batches // plr_interval)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        r1_penalties.append(avg_r1)
        plr_losses.append(avg_plr)
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Generator Loss: {avg_g_loss:.4f}")
        print(f"  Discriminator Loss: {avg_d_loss:.4f}")
        print(f"  R1 Penalty: {avg_r1:.4f}")
        print(f"  Path Length Penalty: {avg_plr:.4f}")
        print(f"  Mean Path Length: {mean_path_length.item():.4f}")

        # print(f"Real scores: min={real_scores.min().item():.3f}, "
        #     f"max={real_scores.max().item():.3f}, mean={real_scores.mean().item():.3f}")
        # print(f"Fake scores: min={fake_scores.min().item():.3f}, "
        #     f"max={fake_scores.max().item():.3f}, mean={fake_scores.mean().item():.3f}")

        current_g_lr = g_optimizer.param_groups[0]['lr']
        current_d_lr = d_optimizer.param_groups[0]['lr']
        # print(f"LR - G: {current_g_lr:.6f}, D: {current_d_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'r1_penalties': r1_penalties,
                'ada_state': ada.state_dict(),  # ADD THIS
                'config': config
            }
            checkpoint_path = os.path.join(save_dir, f'stylegan_checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved locally: {checkpoint_path}")
            
            # Upload to Hugging Face
            try:
                hf_api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo=f"stylegan_checkpoint_epoch_{epoch+1}.pth",
                    repo_id=HF_REPO,
                    token=HF_TOKEN,
                    create_pr=False,
                )
                print(f"✓ Checkpoint uploaded to Hugging Face: epoch {epoch+1}")
            except Exception as e:
                print(f"⚠ Failed to upload checkpoint: {e}")
                print("Continuing training (checkpoint saved locally)")
        
        # Generate samples
        if (epoch + 1) % config["sample_every"] == 0:
            generate_samples(generator, device, epoch + 1, save_dir)
        
        clear_memory()
    
    # Saves final model
    print("\nTraining complete! Saving final model...")
    final_path = os.path.join(save_dir, 'stylegan_final.pth')
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'config': config
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Plots training curves
    plot_training_curves(g_losses, d_losses, r1_penalties, save_dir)
    
    return generator, discriminator, g_losses, d_losses

def plot_training_curves(g_losses, d_losses, r1_penalties, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(g_losses, label='Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(d_losses, label='Discriminator Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(r1_penalties, label='R1 Penalty', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Penalty')
    axes[2].set_title('R1 Regularization')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")

if __name__ == "__main__":
    generator, discriminator, g_losses, d_losses = train_stylegan(
        training_config, 
        checkpoint_path="/content/models--hajar001--StyleGAN-Caleba/snapshots/55ee0021e98714410baa7e8902a82c0f5f639a9c/stylegan_checkpoint_epoch_6.pth"
    )
    print("\n✓ Training completed successfully!")