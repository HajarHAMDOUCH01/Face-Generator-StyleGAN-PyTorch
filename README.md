# StyleGAN-VAE Architecture

A hybrid Variational Autoencoder combining a convolutional encoder with a StyleGAN-inspired decoder for high-quality image generation at 128×128 resolution.

---

## Overview

This architecture merges the probabilistic learning of VAEs with the style-based generation of StyleGAN, enabling both reconstruction and controllable generation through disentangled latent representations.

**Key Features:**
- Progressive style-based decoder with adaptive instance normalization (AdaIN)
- Deep residual encoder with 5 downsampling stages
- Latent space mixing for style disentanglement
- Numerical stability optimizations throughout

---

## Architecture Components

### 1. **Encoder: StrongerEncoder**

Compresses 128×128×3 images into a 512-dimensional latent distribution.

**Structure:**
```
Input: 128×128×3
  ↓ Initial Conv + GroupNorm
128×128×64
  ↓ Down1: Conv + 2×ResidualBlock
64×64×128
  ↓ Down2: Conv + 2×ResidualBlock
32×32×256
  ↓ Down3: Conv + 3×ResidualBlock
16×16×512
  ↓ Down4: Conv + 2×ResidualBlock
8×8×512
  ↓ Down5: Conv + 1×ResidualBlock
4×4×512
  ↓ AdaptiveAvgPool + Flatten
512
  ↓ FC layers
μ(512), log σ²(512)
```

**Features:**
- Residual blocks for gradient flow (11 total)
- GroupNorm for stable training (replaces BatchNorm)
- Kaiming initialization for LeakyReLU activations
- Outputs mean (μ) and log-variance for reparameterization

---

### 2. **Mapping Network**

Transforms latent code **z** into intermediate latent **w** for style control.

**Structure:**
- 8 fully connected layers (512 → 512)
- LeakyReLU(0.2) activations
- Pixel normalization on input z
- Xavier initialization (gain=0.5)

**Purpose:**  
The mapping network disentangles latent factors, making **w** more semantically meaningful than raw **z**.

---

### 3. **Synthesis Network**

Generates images progressively from 4×4 to 128×128 using style modulation.

**Block Structure:**
```
Constant 4×4×512 input
  ↓ StyleLayer (no conv) → StyleLayer (conv)
4×4×512
  ↓ Bilinear Upsample → 2×StyleLayer
8×8×512
  ↓ Bilinear Upsample → 2×StyleLayer
16×16×512
  ↓ Bilinear Upsample → 2×StyleLayer
32×32×512
  ↓ Bilinear Upsample → 2×StyleLayer
64×64×512
  ↓ Bilinear Upsample → 2×StyleLayer
128×128×256
  ↓ ToRGB (1×1 Conv)
128×128×3
```

**Each Synthesis Layer:**
1. **Convolution**: 3×3 spatial feature extraction
2. **Noise Injection**: Adds stochastic variation (learnable weight)
3. **AdaIN (Adaptive Instance Normalization)**:
   - Normalize features: `(x - μ) / σ`
   - Apply style: `y_s * x_normalized + y_b`
   - Where `y_s` (scale) and `y_b` (bias) come from **w** transform
4. **LeakyReLU(0.2)** activation

**Style Modulation:**
- Per-layer **w** vector → Affine transform → (scale, bias)
- Controls feature statistics at each resolution
- Enables fine (high-res) and coarse (low-res) style control

---

### 4. **Style Mixing (Optional)**

During training, combines two latent codes for improved disentanglement:

```python
z1 → w1 (primary style)
z2 → w2 (secondary style)

# Crossover at random layer
w[0:k]   = w1  # Coarse styles (pose, structure)
w[k:end] = w2  # Fine styles (color, texture)
```

**Benefits:**
- Prevents network from assuming correlation between layers
- Improves semantic separation (e.g., face shape vs. hair color)

---

## Mathematical Formulation

### VAE Loss (MSE + VGG19 features distance -from > 20 epoches- + KLD -beta annealing- ) with weights as hyperparameters
```
L = MSE_loss_weight * L_recon + perceptual_loss_weight * percep_distance + β * KL(q(z|x) || p(z)) 

L_recon = ||x - x̂||²  (reconstruction loss)
KL = -0.5 * Σ(1 + log σ² - μ² - σ²)  (KL divergence)
```

### Reparameterization Trick
```
z = μ + σ ⊙ ε,  ε ~ N(0, I)
```
Enables backpropagation through stochastic sampling.

### AdaIN Formula
```
AdaIN(x, w) = y_s * ((x - μ(x)) / σ(x)) + y_b
where (y_s, y_b) = Affine(w)
```

---

## Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Normalization** | GroupNorm (encoder), AdaIN (decoder) | Stable for small batches, style control |
| **Activation** | LeakyReLU(0.2) | Prevents dead neurons, smooth gradients |
| **Upsampling** | Bilinear interpolation | Avoids checkerboard artifacts |
| **Initialization** | Kaiming (conv), Xavier (linear) | Matches activation functions |
| **Clamping** | Throughout (-10, 10) | Prevents NaN/Inf during training |

---

## Usage Modes

### Standard Reconstruction
```python
model = ConvolutionalVAE(z_dim=512, input_size=128, use_style_mixing=False)
recon_x, mu, logvar = model(input_images)
```

### Style Mixing (Training)
```python
model = ConvolutionalVAE(z_dim=512, input_size=128, use_style_mixing=True)
# Requires two latent codes during training
recon_x, mu, logvar = model(input_images)
```

### Generation from Latent
```python
z = torch.randn(batch_size, 512)  # Sample from N(0,I)
generated_images = model.decode(z)
```

---

## Model Statistics

- **Total Synthesis Layers**: 12 (2 per resolution block, 6 blocks)
- **Encoder Depth**: 5 downsampling stages
- **Parameter Count**: ~50-80M (depending on configuration)
- **Input/Output**: 128×128×3 RGB images in [-1, 1]
- **Latent Dimension**: 512 (configurable)

---

## Training Recommendations

**See training_config.py**

---

## References

- **VAE**: Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- **StyleGAN**: Karras et al. (2019) - A Style-Based Generator Architecture
- **AdaIN**: Huang & Belongie (2017) - Arbitrary Style Transfer in Real-time

