import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization (section 4.2 in ProGAN paper)"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate (runtime weight scaling)"""
    def __init__(self, in_features, out_features, bias=True, gain=2):
        super().__init__()
        # Initialize with N(0,1)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # He initialization scale computed at runtime
        self.scale = (gain / in_features) ** 0.5
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """Conv2d layer with equalized learning rate"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=2):
        super().__init__()
        # Initialize with N(0,1)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        # He initialization scale
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = (gain / fan_in) ** 0.5
    
    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class Blur(nn.Module):
    """Blur layer for anti-aliasing (config B in paper)"""
    def __init__(self):
        super().__init__()
        # Binomial filter [1, 2, 1]
        kernel = torch.tensor([[1, 2, 1]], dtype=torch.float32)
        kernel = kernel.t() @ kernel  # 2D separable filter
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))
    
    def forward(self, x):
        # same blur to all channels
        channels = x.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=channels)


class MappingNetwork(nn.Module):
    """Maps latent code z to intermediate latent code w (Figure 1b left)"""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim, gain=2))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        self.pixel_norm = PixelNorm()
    
    def forward(self, z: Tensor) -> Tensor:
        z = self.pixel_norm(z)  
        w = self.mapping(z)
        return w


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (Equation 1 in paper)"""
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        # Learned affine transform "A" from Figure 1b
        self.style_scale = EqualizedLinear(w_dim, channels, gain=1)
        self.style_bias = EqualizedLinear(w_dim, channels, gain=1)
    
    def forward(self, x, w):
        # Normalize to zero mean, unit variance
        x = self.instance_norm(x)
        # Apply learned style transformation
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class NoiseInjection(nn.Module):
    """Stochastic variation through noise (Figure 1b, "B")"""
    def __init__(self, channels):
        super().__init__()
        # Learned per-channel scaling factor
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise


class StyleBlock(nn.Module):
    """Synthesis block with style modulation and noise injection"""
    def __init__(self, in_channels, out_channels, w_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        
        if upsample:
            # Bilinear upsampling + blur for anti-aliasing
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.blur = Blur()
        
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1, gain=2)
        self.noise = NoiseInjection(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.adain = AdaIN(out_channels, w_dim)
    
    def forward(self, x, w, noise: Optional[Tensor] = None):
        if self.upsample:
            x = self.upsample_layer(x)
            x = self.blur(x)
        
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.activation(x)
        x = self.adain(x, w)
        return x


class SynthesisNetwork(nn.Module):
    """Style-based generator synthesis network (Figure 1b right)"""
    def __init__(self, w_dim=512, img_size=128, img_channels=3):
        super().__init__()
        self.w_dim = w_dim
        self.img_size = img_size
        
        # Learned constant input (4×4×512)
        self.const = nn.Parameter(torch.ones(1, 512, 4, 4))
        
        # For 128×128: 4→8→16→32→64→128 (5 resolution levels)
        self.log_size = int(np.log2(img_size))
        self.num_layers = (self.log_size - 1) * 2  # 2 conv layers per resolution
        
        # Channel counts matching paper Table 1 architecture
        # For 128×128: use 512 channels up to 16×16, then reduce
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256,
            64: 128,
            128: 64,
        }
        
        # Initial 4×4 block (2 convolutions, no upsampling)
        self.initial_conv1 = StyleBlock(512, 512, w_dim, upsample=False)
        self.initial_conv2 = StyleBlock(512, 512, w_dim, upsample=False)
        
        # Progressive upsampling blocks
        self.blocks = nn.ModuleList()
        for i in range(3, self.log_size + 1):  # 8×8 to target resolution
            res = 2 ** i
            in_ch = self.channels[res // 2]
            out_ch = self.channels[res]
            
            # First conv with upsampling
            self.blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=True))
            # Second conv without upsampling
            self.blocks.append(StyleBlock(out_ch, out_ch, w_dim, upsample=False))
        
        # Final 1×1 convolution to RGB
        self.to_rgb = EqualizedConv2d(self.channels[img_size], img_channels, kernel_size=1, gain=1)
    
    def forward(self, w):
        batch_size = w.shape[0]
        
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Start from learned constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Apply initial 4×4 convolutions
        layer_idx = 0
        x = self.initial_conv1(x, w[:, layer_idx])
        layer_idx += 1
        x = self.initial_conv2(x, w[:, layer_idx])
        layer_idx += 1
        
        # Progressive upsampling
        for block in self.blocks:
            x = block(x, w[:, layer_idx])
            layer_idx += 1
        
        # Convert to RGB
        rgb = self.to_rgb(x)
        return torch.tanh(rgb)  # [-1, 1] range


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for discriminator"""
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Handle case where batch_size < group_size or not divisible
        if batch_size <= 1:
            # Can't compute stddev with batch of 1, just add zeros
            stddev_channel = torch.zeros(batch_size, 1, height, width, device=x.device)
            return torch.cat([x, stddev_channel], dim=1)
        
        group_size = min(batch_size, self.group_size)
        
        # Only use batches divisible by group_size
        if batch_size % group_size != 0:
            # Trim batch to be divisible
            usable_batch = (batch_size // group_size) * group_size
            x_grouped = x[:usable_batch]
            x_remainder = x[usable_batch:]
        else:
            x_grouped = x
            x_remainder = None
        
        # Compute stddev for grouped portion
        num_groups = x_grouped.shape[0] // group_size
        y = x_grouped.reshape(num_groups, group_size, channels, height, width)
        
        # Compute stddev across group dimension
        y = y - y.mean(dim=1, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=1) + self.eps)
        
        # Average across channels and spatial dimensions
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        
        # Replicate to match spatial dimensions
        y = y.repeat(1, group_size, 1, height, width)
        y = y.reshape(-1, 1, height, width)
        
        # Handle remainder if exists
        if x_remainder is not None:
            # Use last computed stddev for remainder
            remainder_stddev = y[-1:].repeat(x_remainder.shape[0], 1, 1, 1)
            y = torch.cat([y, remainder_stddev], dim=0)
        
        return torch.cat([x, y], dim=1)

class Discriminator(nn.Module):
    """Progressive discriminator (mirrors generator structure)"""
    def __init__(self, img_size=128, img_channels=3):
        super().__init__()
        self.img_size = img_size
        log_size = int(np.log2(img_size))
        
        # Channel progression for 128×128
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256,
            64: 128,
            128: 64,
        }
        
        # FromRGB layer
        self.from_rgb = EqualizedConv2d(img_channels, channels[img_size], kernel_size=1, gain=2)
        
        # Progressive downsampling blocks
        convs = []
        in_ch = channels[img_size]
        
        for i in range(log_size, 2, -1):
            res = 2 ** i
            out_ch = channels[res // 2]
            
            convs.append(EqualizedConv2d(in_ch, in_ch, kernel_size=3, padding=1, gain=2))
            convs.append(nn.LeakyReLU(0.2))
            convs.append(Blur())
            convs.append(EqualizedConv2d(in_ch, out_ch, kernel_size=3, padding=1, gain=2))
            convs.append(nn.LeakyReLU(0.2))
            convs.append(nn.AvgPool2d(2))
            
            in_ch = out_ch
        
        self.convs = nn.Sequential(*convs)
        
        # Final 4×4 block with minibatch stddev
        self.minibatch_stddev = MinibatchStdDev()
        self.final_conv = EqualizedConv2d(in_ch + 1, 512, kernel_size=3, padding=1, gain=2)
        self.final_linear = EqualizedLinear(512 * 4 * 4, 1, gain=1)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.from_rgb(x)
        x = self.activation(x)
        x = self.convs(x)
        
        # Final 4×4 processing
        x = self.minibatch_stddev(x)
        x = self.final_conv(x)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.final_linear(x)
        return x


class StyleGAN(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_size=128, img_channels=3, 
                 mapping_layers=8, style_mixing_prob=0.9):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.style_mixing_prob = style_mixing_prob
        
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        self.synthesis = SynthesisNetwork(w_dim, img_size, img_channels)
        
        self.register_buffer('w_mean', torch.zeros(w_dim))
        self.w_mean_samples = 0
    
    def update_w_mean(self, num_samples=10000):
        with torch.no_grad():
            w_sum = torch.zeros(self.w_dim, device=next(self.parameters()).device)
            for _ in range(0, num_samples, 100):
                z = torch.randn(100, self.z_dim, device=next(self.parameters()).device)
                w = self.mapping(z)
                w_sum += w.sum(dim=0)
            self.w_mean = w_sum / num_samples
            self.w_mean_samples = num_samples
    
    def forward(self, z1: Tensor, z2: Optional[Tensor] = None, return_w: bool = False) -> tuple[Tensor, Optional[Tensor]]:
        w1 = self.mapping(z1)
        
        # Style mixing during training
        if self.training and z2 is not None and torch.rand(1).item() < self.style_mixing_prob:
            w2 = self.mapping(z2)
            
            # Random crossover point
            num_layers = self.synthesis.num_layers
            crossover = torch.randint(1, num_layers, (1,)).item()
            
            w = torch.stack([w1] * num_layers, dim=1)
            w[:, crossover:] = w2.unsqueeze(1)
        else:
            w = w1
        
        rgb = self.synthesis(w)
        
        if return_w:
            return rgb, w
        return rgb, None
    
    @torch.no_grad()
    def generate(self, z: Tensor, truncation_psi: float = 1.0) -> Tensor:
        w = self.mapping(z)
        
        if truncation_psi < 1.0:
            if self.w_mean_samples == 0:
                self.update_w_mean()
            w = self.w_mean + truncation_psi * (w - self.w_mean)
        
        return self.synthesis(w)