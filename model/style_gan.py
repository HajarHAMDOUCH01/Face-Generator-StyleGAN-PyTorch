import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gain=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (gain / in_features) ** 0.5
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = (gain / fan_in) ** 0.5
    
    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class MappingNetwork(nn.Module):
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
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        # Affine transformation A in paper Figure 1b
        self.style_scale = EqualizedLinear(w_dim, channels, gain=1)
        self.style_bias = EqualizedLinear(w_dim, channels, gain=1)
    
    def forward(self, x, w):
        # Normalize
        x = self.instance_norm(x)
        # Apply style
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
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
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        else:
            self.upsample_layer = nn.Identity()
        
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1, gain=2)
        self.noise = NoiseInjection(out_channels) 
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, w, noise: Optional[Tensor] = None):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.activation(x)
        x = self.adain(x, w)
        return x


class SynthesisNetwork(nn.Module):
    """Style-based generator synthesis network (paper section 2, Figure 1b)"""
    def __init__(self, w_dim=512, img_size=128, img_channels=3):
        super().__init__()
        self.w_dim = w_dim
        self.img_size = img_size
        
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Calculate number of layers needed
        self.log_size = int(np.log2(self.img_size))
        self.num_layers = (self.log_size - 1) * 2  # 2 layers per resolution
        
        # Initial 4x4 block (no upsampling)
        self.initial_conv1 = StyleBlock(512, 512, w_dim, upsample=False)
        self.initial_conv2 = StyleBlock(512, 512, w_dim, upsample=False)
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        in_ch = 512
        for i in range(3, self.log_size + 1):  
            out_ch = min(512, 512 // (2 ** max(0, i - 4)))  # Reduce channels at higher resolutions
            # Each resolution has 2 convolutions
            self.blocks.append(StyleBlock(in_ch, out_ch, w_dim, upsample=True))
            self.blocks.append(StyleBlock(out_ch, out_ch, w_dim, upsample=False))
            in_ch = out_ch
        
        # ToRGB (final 1x1 conv to RGB)
        self.to_rgb = EqualizedConv2d(in_ch, img_channels, kernel_size=1, gain=1)
    
    def forward(self, w):
        batch_size = w.shape[0]
        
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Start from learned constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Initial 4x4 block
        layer_idx = 0
        x = self.initial_conv1(x, w[:, layer_idx])
        layer_idx += 1
        x = self.initial_conv2(x, w[:, layer_idx])
        layer_idx += 1
        
        # Progressive blocks
        for block in self.blocks:
            x = block(x, w[:, layer_idx])
            layer_idx += 1
        
        # To RGB
        rgb = self.to_rgb(x)
        return torch.tanh(rgb)  # Output in [-1, 1]


class Discriminator(nn.Module):
    def __init__(self, img_size=128, img_channels=3):
        super().__init__()
        self.img_size = img_size
        log_size = int(np.log2(img_size))
        
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }
        
        convs = []
        in_ch = img_channels
        for i in range(log_size, 2, -1):
            out_ch = channels[2 ** i]
            convs.append(EqualizedConv2d(in_ch, out_ch, kernel_size=3, padding=1, gain=2))
            convs.append(nn.LeakyReLU(0.2))
            convs.append(EqualizedConv2d(out_ch, out_ch, kernel_size=3, padding=1, gain=2))
            convs.append(nn.LeakyReLU(0.2))
            convs.append(nn.AvgPool2d(2))
            in_ch = out_ch
        
        self.convs = nn.Sequential(*convs)
        
        # Final layers at 4x4
        self.final_conv = EqualizedConv2d(in_ch, 512, kernel_size=3, padding=1, gain=2)
        self.final_linear = EqualizedLinear(512 * 4 * 4, 1, gain=1)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        x = self.final_conv(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
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
    
    def forward(self, z1: Tensor, z2: Optional[Tensor] = None, return_w: bool = False) -> tuple[Tensor, Optional[Tensor]]:
        w1 = self.mapping(z1)
        
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
    
    def generate(self, z: Tensor, truncation_psi: float = 1.0) -> Tensor:
        with torch.no_grad():
            w = self.mapping(z)
            if truncation_psi < 1.0:
                # Truncates towards mean w
                w_mean = self.mapping(torch.randn(1000, self.z_dim, device=z.device)).mean(0)
                w = w_mean + truncation_psi * (w - w_mean)
            return self.synthesis(w)