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
        """FIXED: Safer AdaIN with numerical stability"""
        # Instance normalization
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
        # Convolution
        if self.use_conv:
            x = self.conv(x)
            x = torch.clamp(x, -10, 10)
        
        # Add noise
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

    def forward(self, w):
        batch_size = w.size(0)
        
        # Start from constant
        x = self.const.expand(batch_size, -1, -1, -1)
        
        # First block at 4×4
        x = self.layers[0](x, w)
        x = self.layers[1](x, w)
        
        layer_idx = 2
        upsample_idx = 0
        
        # Progressive synthesis
        num_blocks = (self.resolution_log2 - self.start_log2)
        for block in range(num_blocks):
            x = self.upsamples[upsample_idx](x)
            x = self.layers[layer_idx](x, w)
            x = self.layers[layer_idx + 1](x, w)
            
            layer_idx += 2
            upsample_idx += 1
        
        # ToRGB
        rgb = self.to_rgb(x)
        rgb = torch.tanh(rgb)
        
        return rgb


class StyleGANDecoder(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, max_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.max_size = max_size
        
        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=8)
        self.synthesis_network = SynthesisNetwork(w_dim=w_dim, start_size=4, max_size=max_size)
    
    def forward(self, z):
        w = self.mapping_network(z)
        rgb = self.synthesis_network(w)
        return rgb


class ConvolutionalVAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=512, input_size=128):
        super().__init__()
        self.z_dim = z_dim
        self.input_size = input_size
        
        # Encoder
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
        
        self._init_encoder()
        
        h_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.5)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.5)
        nn.init.constant_(self.fc_logvar.bias, -5.0)
        
        self.decoder = StyleGANDecoder(z_dim=z_dim, w_dim=z_dim, max_size=input_size)
    
    def _init_encoder(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -20, 20)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, 1e-6, 10)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = torch.clamp(z, -10, 10)
        return z
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -20, 20)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar