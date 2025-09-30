# Convolutional Variational Autoencoder

![image alt](https://github.com/HajarHAMDOUCH01/Convolutiional_VAE/blob/b4df023fbd80e64d0728bcea3238f3c094c4f1d6/VAE.png)

A PyTorch implementation of a Convolutional Variational Autoencoder for image generation and reconstruction, specifically designed for face datasets.

# Generating a face from sampling from the latent space 

![image alt](https://github.com/HajarHAMDOUCH01/Convolutiional_VAE/blob/7d5bd27816848aca9a0db128e8ec902dbd36c0fb/generated_face.png)

## Features

- **Convolutional Architecture**: Encoder-decoder structure with convolutional and transposed convolutional layers
- **Perceptual Loss**: VGG19-based perceptual loss for improved reconstruction quality
- **Beta Scheduling**: Gradual increase of KL divergence weight during training
- **Checkpointing**: Resume training from saved checkpoints
- **Memory Efficient**: Built-in memory management for GPU training

## Model Architecture

- **Input**: 128×128 RGB images
- **Encoder**: 3 convolutional blocks (32→64→128 channels) with LayerNorm
- **Latent Space**: 256-dimensional latent vector
- **Decoder**: 3 transposed convolutional blocks with skip connections

## Loss Components

1. **Reconstruction Loss**: Binary Cross-Entropy + MAE
2. **KL Divergence Loss**: Regularization term with beta scheduling
3. **Perceptual Loss**: VGG19 feature matching (optional, enabled after epoch 10)

## Quick Start

```python
git clone https://github.com/HajarHAMDOUCH01/Convolutiional_VAE
pip install -r requirements.txt

# Train model
python train_vae.py
```

## Training Configuration

Key parameters in `training_config.py`:
- `batch_size`: 64
- `lr`: 1e-4
- `num_epochs`: 300
- `beta`: KL weight (starts at 0.0, increases every 10 epochs => maximum 1.0)
- `z_dim`: Latent dimension (256)
