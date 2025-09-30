import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.compile
def pixel_norm(z, epsilon=1e-8):
    return z * torch.rsqrt(torch.mean(z*z, dim=1, keepdim=True) + epsilon)


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
            layers.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim, bias=True))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping_network = nn.Sequential(*layers)
        
    def forward(self, z):
        z_normalized = pixel_norm(z)
        w = self.mapping_network(z_normalized)
        return w


class WTransform(nn.Module):
    """
    Learned affine transform A that maps w ∈ W to style parameters (y_s, y_b)
    """
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.channels = channels
        self.affine = nn.Linear(w_dim, 2 * channels)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.affine.weight, mode='fan_in', nonlinearity='linear')
        with torch.no_grad():
            self.affine.bias.data[:self.channels] = 1  # bias for y_s (scale)
            self.affine.bias.data[self.channels:] = 0  # bias for y_b (shift)

    def forward(self, w):
        style = self.affine(w)
        y_s, y_b = style.chunk(2, dim=1)
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
            
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
            
        self.w_transform = WTransform(out_channels, w_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activation = nn.LeakyReLU(0.2)
        
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def _get_noise(self, batch_size):
        return torch.randn(batch_size, 1, self.size, self.size, device=self.device)
    
    def adain(self, x, y_s, y_b):
        # Instance normalization
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-8
        x_normalized = (x - mean) / std
        
        # Apply style
        y_s = y_s[:, :, None, None]
        y_b = y_b[:, :, None, None]
        
        return x_normalized * y_s + y_b
    
    def forward(self, x, w):
        # Convolution
        if self.use_conv:
            x = self.conv(x)
        
        # Add noise
        noise = self._get_noise(x.size(0))
        x = x + noise * self.noise_weight
        
        # AdaIN
        y_s, y_b = self.w_transform(w)
        x = self.adain(x, y_s, y_b)
        
        # Activation
        x = self.activation(x)
        
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
        
        # Learnable constant input
        self.const = nn.Parameter(torch.randn(1, 512, start_size, start_size))
        
        # Build synthesis layers
        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # First block at 4×4 
        in_ch = 512
        out_ch = 512
        self.layers.append(SynthesisLayer(in_ch, out_ch, w_dim, start_size, use_conv=False))
        self.layers.append(SynthesisLayer(out_ch, out_ch, w_dim, start_size, use_conv=True))
        
        # Progressive blocks: 8×8, 16×16, 32×32, 64×64, 128×128
        current_size = start_size
        for res_log2 in range(self.start_log2 + 1, self.resolution_log2 + 1):
            current_size *= 2
            in_ch = out_ch
            out_ch = get_channels(res_log2)
            
            # Upsample
            self.upsamples.append(BilinearUpsample(scale_factor=2))
            
            # Two synthesis layers per resolution
            self.layers.append(SynthesisLayer(in_ch, out_ch, w_dim, current_size, use_conv=True))
            self.layers.append(SynthesisLayer(out_ch, out_ch, w_dim, current_size, use_conv=True))
        
        self.to_rgb = nn.Conv2d(out_ch, 3, kernel_size=1)

    def forward(self, w):
        """
        Args:
            w: latent code in W space, shape (batch_size, w_dim)
        Returns:
            RGB image, shape (batch_size, 3, max_size, max_size)
        """
        batch_size = w.size(0)
        
        # the start is from constant
        x = self.const.expand(batch_size, -1, -1, -1)
        
        # First block at 4×4
        x = self.layers[0](x, w)
        x = self.layers[1](x, w)
        
        layer_idx = 2
        upsample_idx = 0
        
        # Progressive synthesis
        num_blocks = (self.resolution_log2 - self.start_log2)
        for block in range(num_blocks):
            # Upsample
            x = self.upsamples[upsample_idx](x)
            
            # Two layers per block
            x = self.layers[layer_idx](x, w)
            x = self.layers[layer_idx + 1](x, w)
            
            layer_idx += 2
            upsample_idx += 1
        
        rgb = self.to_rgb(x)
        
        return rgb


class StyleGANDecoder(nn.Module):
    """
    StyleGAN-based decoder: Mapping network f: Z → W, Synthesis network g: W → Image
    """
    def __init__(self, z_dim=512, w_dim=512, max_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.max_size = max_size
        
        # Mapping network: Z → W
        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=8)
        
        # Synthesis network: W → Image
        self.synthesis_network = SynthesisNetwork(w_dim=w_dim, start_size=4, max_size=max_size)
    
    def forward(self, z):
        """
        Args:
            z: latent code in Z space, shape (batch_size, z_dim)
        Returns:
            RGB image, shape (batch_size, 3, max_size, max_size)
        """
        w = self.mapping_network(z)
        rgb = self.synthesis_network(w)
        return rgb


class ConvolutionalVAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=512, input_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.input_size = input_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128→64
            nn.LayerNorm([32, 64, 64]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64→32
            nn.LayerNorm([64, 32, 32]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32→16
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16→8
            nn.LayerNorm([256, 8, 8]),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        h_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        # Decoder: StyleGAN
        self.decoder = StyleGANDecoder(z_dim=z_dim, w_dim=z_dim, max_size=input_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        recon_x = torch.tanh(recon_x)  # Output in [-1, 1]
        return recon_x, mu, logvar

