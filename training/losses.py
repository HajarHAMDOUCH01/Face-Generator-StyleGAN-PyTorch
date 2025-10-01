import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_divergence_loss(mu, logvar):
    """
    Standard KL Divergence: KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()  


def kl_divergence_loss_with_free_bits(mu, logvar, free_bits=0.5):
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    kld_per_dim = torch.clamp(kld_per_dim, min=free_bits)
    
    kld = torch.sum(kld_per_dim, dim=1).mean()
    
    return kld


def reconstruction_loss_mse(recon_x, x):
    """
    MSE reconstruction loss (better for [-1, 1] range images)
    StyleGAN outputs are in [-1, 1] range, so MSE is more appropriate than BCE
    """
    return F.mse_loss(recon_x, x, reduction='mean')


def reconstruction_loss_mae(recon_x, x):
    """
    MAE (L1) reconstruction loss - often more robust than MSE
    """
    return F.l1_loss(recon_x, x, reduction='mean')


VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        from torchvision.models import vgg19, VGG19_Weights
        vgg_features = vgg19(weights='DEFAULT').features

        self.vgg_model_weights = VGG19_Weights
        
        # VGG expects [0, 1] range, normalization inside forward()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Extract features at different layers
        # conv1_2 (early features)
        self.slice1 = nn.Sequential(*[vgg_features[x] for x in range(0, 4)])  
        # conv2_2 (mid-level features)
        self.slice2 = nn.Sequential(*[vgg_features[x] for x in range(4, 9)])  
        # conv3_4 (higher-level features)
        self.slice3 = nn.Sequential(*[vgg_features[x] for x in range(9, 18)])
        # conv4_4 (semantic features)
        self.slice4 = nn.Sequential(*[vgg_features[x] for x in range(18, 27)])
            
        for param in self.parameters():
            param.requires_grad = False
    
    def normalize_image(self, x):
        # [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        # Normalization with ImageNet stats
        x = (x - self.mean) / self.std
        return x
            
    def forward(self, x):
        """
        Args:
            x: image in [-1, 1] range
        Returns:
            List of feature maps at different layers
        """
        x = self.normalize_image(x)
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        return [h1, h2, h3, h4]


def perceptual_loss_vae(vgg19_model, recon_x, x):
    layer_weights = [0.0625, 0.125, 0.25, 0.5]
    
    total_loss = 0.0
    
    with torch.no_grad():
        x_features = vgg19_model(x)
    
    recon_features = vgg19_model(recon_x)
    
    for i, (x_feat, recon_feat) in enumerate(zip(x_features, recon_features)):
        layer_loss = F.mse_loss(recon_feat, x_feat, reduction='mean')
        
        weighted_loss = layer_weights[i] * layer_loss
        total_loss += weighted_loss
    
    return total_loss


def vae_loss_with_perceptual(vgg19_model, recon_x, x, mu, logvar, beta=1.0, 
                             mse_weight=1.0, percep_weight=0.1, use_percep=True,
                             free_bits=None):
    mse_loss = mse_weight * reconstruction_loss_mse(recon_x, x)
    
    if use_percep:
        percep_loss = percep_weight * perceptual_loss_vae(vgg19_model, recon_x, x)
        recon_loss = mse_loss + percep_loss
    else:
        recon_loss = mse_loss
    
    if free_bits is not None and free_bits > 0:
        kld_loss = kl_divergence_loss_with_free_bits(mu, logvar, free_bits=free_bits)
    else:
        kld_loss = kl_divergence_loss(mu, logvar)
    
    total_loss = recon_loss + beta * kld_loss
    
    return total_loss, beta * kld_loss, recon_loss


def vae_loss_simple(recon_x, x, mu, logvar, beta=1.0, free_bits=None):
    recon_loss = reconstruction_loss_mse(recon_x, x)
    
    if free_bits is not None and free_bits > 0:
        kld_loss = kl_divergence_loss_with_free_bits(mu, logvar, free_bits=free_bits)
    else:
        kld_loss = kl_divergence_loss(mu, logvar)
    
    total_loss = recon_loss + beta * kld_loss
    
    return total_loss, beta * kld_loss, recon_loss