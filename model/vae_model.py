import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.compile
def pixel_norm(z, epsilon=1e-8):
    return z * torch.rsqrt(torch.mean(z*z, dim=1, keepdim=True) + epsilon)



class ResidualBlock(nn.Module):
    """Residual block for encoder - adds capacity without depth explosion"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)  
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = x + residual  
        x = self.activation(x)
        return x


class StrongerEncoder(nn.Module):
    """
    Much deeper encoder with residual connections
    Goes from 3 channels → 512 channels with 5 downsampling stages
    """
    def __init__(self, z_dim=512):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2)
        )
        
        # 128×128 → 64×64
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # 64×64 → 32×32
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # 32×32 → 16×16
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)  
        )
        
        # 16×16 → 8×8
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # 8×8 → 4×4
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.initial(x)   # 128×128×64
        x = self.down1(x)      # 64×64×128
        x = self.down2(x)      # 32×32×256
        x = self.down3(x)      # 16×16×512
        x = self.down4(x)      # 8×8×512
        x = self.down5(x)      # 4×4×512
        
        x = self.pool(x)       # 1×1×512
        x = self.flatten(x)    # 512
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class BilinearUpsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            linear = nn.Linear(z_dim if i == 0 else w_dim, w_dim, bias=True)
            nn.init.xavier_normal_(linear.weight, gain=0.5)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))
        self.mapping_network = nn.Sequential(*layers)
        
    def forward(self, z):
        z_normalized = pixel_norm(z)
        w = self.mapping_network(z_normalized)
        w = torch.clamp(w, -10, 10)
        return w


class WTransform(nn.Module):
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.channels = channels
        self.affine = nn.Linear(w_dim, 2 * channels)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.affine.weight, gain=0.5)
        with torch.no_grad():
            self.affine.bias.data[:self.channels] = 1.0
            self.affine.bias.data[self.channels:] = 0.0

    def forward(self, w):
        style = self.affine(w)
        y_s, y_b = style.chunk(2, dim=1)
        y_s = torch.clamp(y_s, 0.1, 10.0)
        y_b = torch.clamp(y_b, -10, 10)
        return y_s, y_b


class SynthesisLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, size, use_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.size = size
        self.use_conv = use_conv

        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                  stride=1, padding=1)
            nn.init.xavier_normal_(self.conv.weight, gain=0.5)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
            
        self.w_transform = WTransform(out_channels, w_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activation = nn.LeakyReLU(0.2)
        
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def _get_noise(self, batch_size):
        return torch.randn(batch_size, 1, self.size, self.size, device=self.device)
    
    def adain(self, x, y_s, y_b, eps=1e-8):
        """AdaIN with numerical stability"""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        
        x_normalized = (x - mean) / std
        x_normalized = torch.clamp(x_normalized, -10, 10)
        
        y_s = y_s[:, :, None, None]
        y_b = y_b[:, :, None, None]
        
        out = x_normalized * y_s + y_b
        out = torch.clamp(out, -10, 10)
        return out
    
    def forward(self, x, w):
        if self.use_conv:
            x = self.conv(x)
            x = torch.clamp(x, -10, 10)
        
        noise = self._get_noise(x.size(0))
        x = x + noise * self.noise_weight
        
        y_s, y_b = self.w_transform(w)
        x = self.adain(x, y_s, y_b)
        x = self.activation(x)
        x = torch.clamp(x, -10, 10)
        
        return x


class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim=512, start_size=4, max_size=128):
        super().__init__()
        self.w_dim = w_dim
        self.start_size = start_size
        self.max_size = max_size

        self.resolution_log2 = int(torch.log2(torch.tensor(max_size)).item())
        self.start_log2 = int(torch.log2(torch.tensor(start_size)).item())
        
        def get_channels(res):
            base_channels = 512
            return min(base_channels, 512 // (2 ** max(0, res - 4)))
        
        self.const = nn.Parameter(torch.randn(1, 512, start_size, start_size) * 0.1)
        
        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # First block at 4×4
        in_ch = 512
        out_ch = 512
        self.layers.append(SynthesisLayer(in_ch, out_ch, w_dim, start_size, use_conv=False))
        self.layers.append(SynthesisLayer(out_ch, out_ch, w_dim, start_size, use_conv=True))
        
        # Progressive blocks
        current_size = start_size
        for res_log2 in range(self.start_log2 + 1, self.resolution_log2 + 1):
            current_size *= 2
            in_ch = out_ch
            out_ch = get_channels(res_log2)
            
            self.upsamples.append(BilinearUpsample(scale_factor=2))
            self.layers.append(SynthesisLayer(in_ch, out_ch, w_dim, current_size, use_conv=True))
            self.layers.append(SynthesisLayer(out_ch, out_ch, w_dim, current_size, use_conv=True))
        
        # ToRGB
        self.to_rgb = nn.Conv2d(out_ch, 3, kernel_size=1)
        nn.init.xavier_normal_(self.to_rgb.weight, gain=0.1)
        nn.init.zeros_(self.to_rgb.bias)
        
        self.num_style_layers = len(self.layers)

    def forward(self, w):
        batch_size = w.size(0)
        
        if w.dim() == 2:  # [B, w_dim] - broadcast to all layers
            w = w.unsqueeze(1).expand(-1, self.num_style_layers, -1)  # [B, num_layers, w_dim]
        
        # Start from constant
        x = self.const.expand(batch_size, -1, -1, -1)
        
        # First block at 4×4
        x = self.layers[0](x, w[:, 0])  # Use layer-specific W
        x = self.layers[1](x, w[:, 1])
        
        layer_idx = 2
        upsample_idx = 0
        
        num_blocks = (self.resolution_log2 - self.start_log2)
        for block in range(num_blocks):
            x = self.upsamples[upsample_idx](x)
            x = self.layers[layer_idx](x, w[:, layer_idx])      # Per-layer W
            x = self.layers[layer_idx + 1](x, w[:, layer_idx + 1])
            
            layer_idx += 2
            upsample_idx += 1
        
        # ToRGB
        rgb = self.to_rgb(x)
        rgb = torch.tanh(rgb)
        
        return rgb


class StyleGANDecoder(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, max_size=128, use_style_mixing=True):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.max_size = max_size
        self.use_style_mixing = use_style_mixing
        
        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=8)
        self.synthesis_network = SynthesisNetwork(w_dim=w_dim, start_size=4, max_size=max_size)
        
        # Number of synthesis layers
        self.num_style_layers = self.synthesis_network.num_style_layers
    
    def forward(self, z, return_w=False):
        batch_size = z.size(0)
        
        # Map Z to W
        w = self.mapping_network(z)  # [B, w_dim]
        
        w_broadcast = w.unsqueeze(1).expand(-1, self.num_style_layers, -1)  
        
        # Generate image
        rgb = self.synthesis_network(w_broadcast)
        
        if return_w:
            return rgb, w_broadcast
        return rgb


class StyleGANDecoderWithMixing(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, max_size=128, mixing_prob=0.5):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.max_size = max_size
        self.mixing_prob = mixing_prob   
        
        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=8)
        self.synthesis_network = SynthesisNetwork(w_dim=w_dim, start_size=4, max_size=max_size)
        
        self.num_style_layers = self.synthesis_network.num_style_layers
    
    def forward(self, z, z2=None):
        batch_size = z.size(0)
        
        # Map primary Z to W
        w1 = self.mapping_network(z)  # [B, w_dim]
        
        if self.training and z2 is not None and torch.rand(1).item() < self.mixing_prob:
            # Map secondary Z to W
            w2 = self.mapping_network(z2)  # [B, w_dim]
            
            # Choose random crossover point
            crossover_layer = torch.randint(1, self.num_style_layers, (1,)).item()
            
            # Create mixed W: use w1 for early layers, w2 for later layers
            w = torch.zeros(batch_size, self.num_style_layers, self.w_dim, device=z.device)
            w[:, :crossover_layer] = w1.unsqueeze(1)
            w[:, crossover_layer:] = w2.unsqueeze(1)
        else:
            # No mixing: broadcast w1 to all layers
            w = w1.unsqueeze(1).expand(-1, self.num_style_layers, -1)
        
        # Generate image
        rgb = self.synthesis_network(w)
        return rgb


class ConvolutionalVAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=512, input_size=128, use_style_mixing=False):
        super().__init__()
        self.z_dim = z_dim
        self.input_size = input_size
        self.use_style_mixing = use_style_mixing
        
        self.encoder = StrongerEncoder(z_dim=z_dim)
        
        if use_style_mixing:
            self.decoder = StyleGANDecoderWithMixing(
                z_dim=z_dim, 
                w_dim=z_dim, 
                max_size=input_size,
                mixing_prob=0.5
            )
        else:
            self.decoder = StyleGANDecoder(
                z_dim=z_dim, 
                w_dim=z_dim, 
                max_size=input_size,
                use_style_mixing=True
            )
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -20, 20)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 10)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = torch.clamp(z, -10, 10)
        return z
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        
        mu = torch.clamp(mu, -5, 5)
        logvar = torch.clamp(logvar, -10, 2)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar