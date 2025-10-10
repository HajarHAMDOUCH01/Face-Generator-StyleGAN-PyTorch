import torch
import torch.nn.functional as F
from typing import Optional

class AugmentationPipeline:
    """
    Implements augmentations.
    Categories: pixel blitting, geometric, color transforms.
    """
    def __init__(self, device: torch.device):
        self.device = device
        
    def apply(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """
        Apply augmentations with probability p per transformation.
        """
        if p == 0:
            return x
        
        x = x.clone()
        
        B, C, H, W = x.shape
        original_h, original_w = H, W  # Save original dimensions
        
        # Horizontal flip (per-sample)
        flip_mask = torch.rand(B, device=self.device) < p
        if flip_mask.any():
            x_flipped = torch.flip(x, dims=[3])
            flip_mask_expanded = flip_mask.view(B, 1, 1, 1).expand_as(x)
            x = torch.where(flip_mask_expanded, x_flipped, x)
        
        # 90-degree rotation
        if torch.rand(1).item() < p:
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k=k, dims=[2, 3])
            # After rotation, swap H and W if k is odd
            if k % 2 == 1:
                original_h, original_w = original_w, original_h
        
        # Integer translation
        if torch.rand(1).item() < p:
            max_shift = x.shape[2] // 8
            tx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            ty = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            x = torch.roll(x, shifts=(ty, tx), dims=(2, 3))
        
        # Isotropic scaling 
        if torch.rand(1).item() < p:
            scale = torch.exp(torch.randn(1) * 0.2).item()
            scale = max(0.5, min(2.0, scale))
            
            if abs(scale - 1.0) > 0.01:  
                h, w = x.shape[2], x.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                if scale > 1.0:  
                    crop_top = (new_h - h) // 2
                    crop_left = (new_w - w) // 2
                    x = x[:, :, crop_top:crop_top+h, crop_left:crop_left+w]
                else:  # Pad to exact size
                    pad_h_total = h - new_h
                    pad_w_total = w - new_w
                    pad_top = pad_h_total // 2
                    pad_bottom = pad_h_total - pad_top
                    pad_left = pad_w_total // 2
                    pad_right = pad_w_total - pad_left
                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        
        # Rotation (continuous, small angles)
        if torch.rand(1).item() < p:
            angle = torch.randn(1).item() * 10
            theta = torch.tensor([
                [torch.cos(torch.tensor(angle * 3.14159 / 180)), 
                -torch.sin(torch.tensor(angle * 3.14159 / 180)), 0],
                [torch.sin(torch.tensor(angle * 3.14159 / 180)), 
                torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
            ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
            
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        # Color transforms
        if torch.rand(1).item() < p:
            brightness = torch.randn(1).item() * 0.2
            x = x + brightness
        
        if torch.rand(1).item() < p:
            contrast = torch.exp(torch.randn(1) * 0.5).item()
            x = (x - x.mean(dim=[2, 3], keepdim=True)) * contrast + x.mean(dim=[2, 3], keepdim=True)
        
        if torch.rand(1).item() < p:
            hue_shift = torch.randn(1, 1, 1, 1, device=x.device) * 0.1
            x = x + hue_shift
        
        if torch.rand(1).item() < p:
            sat_factor = torch.exp(torch.randn(1) * 0.5).item()
            gray = x.mean(dim=1, keepdim=True)
            x = gray + (x - gray) * sat_factor
        
        x = torch.clamp(x, -1, 1)
        
        assert x.shape[2] == original_h and x.shape[3] == original_w, \
            f"Augmentation changed dimensions! Expected [{original_h}, {original_w}], got [{x.shape[2]}, {x.shape[3]}]"
        
        return x
    
class ADAugment:
    """
    Adaptive Discriminator Augmentation controller.
    """
    def __init__(
        self,
        target_rt: float = 0.6, 
        adjustment_speed_imgs: int = 500_000,  
        batch_size: int = 64,
        initial_p: float = 0.0,
        update_interval: int = 4, 
        augment_pipe: Optional[AugmentationPipeline] = None,
        device: torch.device = torch.device('cuda')
    ):
        self.target_rt = target_rt
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.device = device
        
        # Calculate adjustment step
        # p goes from 0→1 in adjustment_speed_imgs images
        # Update happens every (update_interval * batch_size) images
        imgs_per_update = update_interval * batch_size
        num_updates_to_max = adjustment_speed_imgs / imgs_per_update
        self.adjustment_step = 1.0 / num_updates_to_max
        
        self._p = initial_p
        self.augment_pipe = augment_pipe or AugmentationPipeline(device)
        
        # Rolling buffer for real_logits (store last N minibatches)
        self.logits_buffer = []
        self.batch_counter = 0
        
        # Statistics tracking
        self.rt_history = []
        self.p_history = []
    
    @property
    def p(self) -> float:
        return max(0.0, min(self._p, 0.8))

    
    def apply(self, images: torch.Tensor, p: Optional[float] = None) -> torch.Tensor:
        """Apply augmentations to images"""
        p_used = p if p is not None else self._p
        return self.augment_pipe.apply(images, p_used)
    
    def update(self, real_logits: torch.Tensor) -> float:
        """
        Update p based on discriminator outputs on real images.
        
        Args:
            real_logits: Raw discriminator outputs for real images [B]
        
        Returns:
            Current p value
        """
        # Accumulate logits
        self.logits_buffer.append(real_logits.detach())
        self.batch_counter += 1
        
        # Update every `update_interval` minibatches
        if self.batch_counter >= self.update_interval:
            # Compute r_t = E[sign(D_train)] over last N batches
            # From Section 3, Equation 1, page 5
            all_logits = torch.cat(self.logits_buffer, dim=0)
            rt = torch.sign(all_logits).mean().item()
            
            # Adjust p
            if rt > self.target_rt:
                # Too much confidence → increase augmentation
                self._p += self.adjustment_step
            else:
                # Too little confidence → decrease augmentation
                self._p -= self.adjustment_step
            
            # Clamp p to [0, 0.85]
            # From Section 2.2, Figure 3b: "p remains below ~0.85, the generated images are always oriented correctly"
            self._p = max(0.0, min(0.85, self._p))
            
            # Track statistics
            self.rt_history.append(rt)
            self.p_history.append(self._p)
            
            # Reset buffer
            self.logits_buffer = []
            self.batch_counter = 0
        
        return self._p
    
    def state_dict(self) -> dict:
        return {
            'p': self._p,
            'batch_counter': self.batch_counter,
            'logits_buffer': [l.cpu() for l in self.logits_buffer],
            'rt_history': self.rt_history,
            'p_history': self.p_history,
        }
    
    def load_state_dict(self, state: dict):
        self._p = state['p']
        self.batch_counter = state['batch_counter']
        self.logits_buffer = [l.to(self.device) for l in state['logits_buffer']]
        self.rt_history = state['rt_history']
        self.p_history = state['p_history']